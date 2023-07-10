#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@original-author: GustavZ
@author: Club SONIA
"""

import os
import sys
import cv2
import copy
import json
import yaml
import rospy
import time
import _thread

import numpy as np
import tensorflow as tf

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage as SensorImage
from std_msgs.msg import Empty
from sonia_common.msg import DetectionArray, Detection, ChangeNetworkMsg, BoundingBox2D

from datetime import datetime
from object_detection.utils import label_map_util
from stuff.helper import FPS2, SessionWorker
from threading import Lock

try:
    from yaml import CLoader as Loader
except:
    from yaml import Loader

tensorrtEnabled = False
try:
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    tensorrtEnabled = True
except Exception as err:
    print(err)


class ObjectDetection:
    def __init__(self):
        rospy.init_node('proc_detection')
        rospy.loginfo("found gpu: {}".format(tf.test.gpu_device_name()))

        if (tensorrtEnabled):
            rospy.loginfo("Using tensorrt!")
        else:
            rospy.loginfo("cannot import tensorrt, probably because the gpu is not detected!")

        self.cv_bridge = CvBridge()
        self.fps_interval = 5
        self.detection_mutex = Lock()

        self.sess = None
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.num_detections = None

        self.image_subscriber = None
        self.prev_model = None
        self.initial_model = None
        self.fps_limit = None
        self.run_with_tensorrt = None
        self.trt_precision_mode = None
        self.trt_segment_size = None
        self.trt_image_width = None
        self.trt_image_height = None
        self.frame = None

        self.get_config()

        self.network_subscriber = rospy.Subscriber("/proc_detection/change_network", ChangeNetworkMsg,
                                                   self.handle_change_network)
        self.network_publisher = rospy.Publisher("/proc_detection/status", ChangeNetworkMsg, queue_size=10)
        self.stop_subscriber = rospy.Subscriber("/proc_detection/stop_topic", Empty, self.stop_topic)
        self.bbox_publisher = rospy.Publisher('/proc_detection/bounding_box', DetectionArray, queue_size=10)
        self.detection_mutex.acquire()
        self.detection_graph = self.load_frozen_model(self.initial_model)
        self.detection_mutex.release()
        self.yolo_classes = [
            "Bins_Abydos_1", "Bins_Abydos_2", "Bins_Earth_1", "Bins_Earth_2", "Gate_Abydos", "Gate_Earth",
            "Glyph_Abydos_1",
            "Glyph_Abydos_2", "Glyph_Earth_1", "Glyph_Earth_2", "Stargate_Closed", "Stargate_Open"
        ]
        _thread.start_new_thread(self.detection, ())
        rospy.spin()

    def image_msg_callback(self, img):
        self.frame = self.cv_bridge.compressed_imgmsg_to_cv2(img, desired_encoding="bgr8")

    def load_graph_def(self, filepath):
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(filepath, "rb") as f:
            serialized_graph = f.read()
        graph_def.ParseFromString(serialized_graph)
        return graph_def

    def load_frozen_model(self, model_name):
        if (model_name != self.prev_model and model_name != "@default"):
            self.prev_model = model_name
            rospy.loginfo("load a new frozen model {}".format(model_name))
            model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "external", 'models', model_name,
                                     'saved_model')
            detection_function = tf.function()
            if tensorrtEnabled and self.run_with_tensorrt:
                output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "external", 'models', model_name,
                                          'trt')

                if not os.path.exists(output_dir):
                    params = trt.DEFAULT_TRT_CONVERSION_PARAMS
                    params = params._replace(precision_mode=self.trt_precision_mode)
                    converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_dir, conversion_params=params)

                    matrix_type = np.float32

                    if self.trt_precision_mode == "FP32":
                        matrix_type = np.float32
                    elif self.trt_precision_mode == "FP16":
                        matrix_type = np.float16
                    elif self.trt_precision_mode == "INT8":
                        matrix_type = np.uint8

                    if self.trt_precision_mode == "INT8":
                        def calibration_input_fn():
                            inp1 = np.random.normal(size=(1, 1, self.trt_image_width, self.trt_image_height, 3)).astype(
                                matrix_type)
                            yield inp1

                        converter.convert(calibration_input_fn=calibration_input_fn)
                    else:
                        converter.convert()

                    def input_fn():
                        inp1 = np.random.normal(size=(1, 1, self.trt_image_width, self.trt_image_height, 3)).astype(
                            matrix_type)
                        yield inp1

                    converter.build(input_fn=input_fn)
                    converter.save(output_dir)

                loaded_model = tf.saved_model.load(output_dir, tags=['serve'])
            else:
                loaded_model = tf.saved_model.load(model_dir)
                loaded_model = loaded_model.signatures["serving_default"]

            rospy.loginfo("model " + model_name + " is loaded")

            return loaded_model

        else:

            rospy.loginfo("keep the previous model " + model_name)
            return self.detection_graph

    def stop_topic(self, data):
        if self.image_subscriber is not None:
            self.image_subscriber.unregister()
            self.image_subscriber = None

    def handle_change_network(self, data):
        if self.image_subscriber is not None:
            self.image_subscriber.unregister()
            self.image_subscriber = None

        if data.network_name != self.prev_model:
            self.detection_mutex.acquire()
            self.detection_graph = self.load_frozen_model(data.network_name)
            self.detection_mutex.release()

        self.image_subscriber = rospy.Subscriber(data.topic, SensorImage, self.image_msg_callback)
        self.threshold = data.threshold / 100.0
        self.fps = FPS2(self.fps_interval).start()

    def run_inference_for_single_image(self, model, img):
        img = np.asarray(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sq_img = np.zeros((640, 640, 3), dtype=np.uint8)
        sq_img[120:520, 20:620, :] = img
        sq_img = np.transpose(sq_img, (2, 0, 1))
        sq_img = np.expand_dims(sq_img, axis=0)
        sq_img = sq_img / 255.0
        tensor = tf.convert_to_tensor(sq_img.astype(np.float32))
        output = model(images=tensor)["output0"][0].numpy()
        output = output.transpose()
        boxes = [row for row in [self.parse_row(row) for row in output] if row[5] > self.threshold]
        boxes.sort(key=lambda x: x[5], reverse=True)
        result = []
        while len(boxes) > 0:
            result.append(boxes[0])
            boxes = [box for box in boxes if self.iou(box, boxes[0]) < 0.7]
        if len(result) != 0:
            rospy.loginfo(len(result))
            rospy.loginfo(result)
        return result

    def detection(self):
        rospy.loginfo("detection thread launched!")
        while not rospy.is_shutdown():
            statusmsg = ChangeNetworkMsg()
            if self.image_subscriber is None:
                statusmsg.topic = "None"
            else:
                statusmsg.topic = self.image_subscriber.name
            statusmsg.network_name = self.prev_model
            self.network_publisher.publish(statusmsg)
            if self.frame is not None:

                image = self.frame
                self.frame = None
                image.setflags(write=1)

                self.detection_mutex.acquire()
                output_list = self.run_inference_for_single_image(self.detection_graph, image)
                self.detection_mutex.release()

                list_detection = DetectionArray()

                for output in output_list:
                    detection = Detection()
                    detection.top = min(max(output[0] - 120, 0), 400) / 400
                    detection.left = min(max(output[1] - 20, 0), 600) / 600
                    detection.bottom = min(max(output[2] - 120, 0), 400) / 400
                    detection.right = min(max(output[3] - 20, 0), 600) / 600

                    detection.confidence = output[5]
                    detection.class_name = output[4]

                    list_detection.detected_object.append(detection)

                self.bbox_publisher.publish(list_detection)
                self.fps.update()
            else:
                time.sleep(0.01)

    def get_config(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'external', 'config', 'config.yml'),
                  'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=Loader)

        self.initial_model = cfg['initial_model']
        self.fps_limit = cfg['fps_limit']
        self.num_classes = cfg['max_num_classes']
        self.run_with_tensorrt = cfg['run_with_tensorrt']
        self.trt_precision_mode = cfg['trt_precision_mode']
        self.trt_segment_size = cfg['trt_segment_size']
        self.trt_image_width = cfg['trt_image_width']
        self.trt_image_height = cfg['trt_image_height']

    def __del__(self):
        self.network_subscriber.unregister()
        self.stop_subscriber.unregister()

    def parse_row(self, row):
        xc, yc, w, h = row[:4]
        x1 = (xc - w / 2)
        y1 = (yc - h / 2)
        x2 = (xc + w / 2)
        y2 = (yc + h / 2)
        prob = row[4:].max(initial=0)
        class_id = row[4:].argmax()
        label = self.yolo_classes[class_id]
        return [x1, y1, x2, y2, label, prob]

    def intersection(self, box1, box2):
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        return (x2 - x1) * (y2 - y1)

    def union(self, box1, box2):
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        return box1_area + box2_area - self.intersection(box1, box2)

    def iou(self, box1, box2):
        return self.intersection(box1, box2) / self.union(box1, box2)


if __name__ == '__main__':
    ObjectDetection()
