#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@original-author: GustavZ
@author: Club SONIA
"""
from datetime import datetime
import numpy as np
import os
import tensorflow as tf
import copy
import yaml
from cv_bridge import CvBridge
from tensorflow.core.framework import graph_pb2
import rospy
from sensor_msgs.msg import CompressedImage as SensorImage
from sonia_common.msg import DetectionArray, Detection, BoundingBox2D
from geometry_msgs.msg import Pose2D
import json
from sonia_common.srv import ChangeNetwork, ChangeNetworkRequest, ChangeNetworkResponse
import sys
import tensorflow.contrib.tensorrt as trt

import cv2

# Protobuf Compilation (once necessary)
# os.system('protoc object_detection/protos/*.proto --python_out=.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from stuff.helper import FPS2, SessionWorker

import time
import thread


class ObjectDetection:
    def __init__(self):
        rospy.init_node('proc_detection')

        self.frame = None
        self.cv_bridge = CvBridge()
        self.category_index = None
        self.detection_graph = None
        self.score = None
        self.expand = None
        self.gpu_worker = None
        self.cpu_worker = None
        self.gpu_opts = None
        self.cpu_opts = None
        self.fps = None
        self.image_tensor = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.num_detections = None
        self.score_in = None
        self.expand_in = None
        self.sess = None
        self.cur_frames = 0
        self.boxes = None
        self.classes = None
        self.scores = None
        self.topic_subscriber = None
        self.detection_thresh = None
        self.num_classes=None
        self.image_subscriber = None
        self.model = None
        self.prev_model = None

        self.get_config()
        self.model_path = None
        self.label_path = None
        self.finish_init = False


        self.network_service = rospy.Service("proc_detection/change_network", ChangeNetwork, self.handle_change_network)
        #self.image_publisher = rospy.Publisher(self.topic_publisher, SensorImage, queue_size=1)
        self.bbox_publisher = rospy.Publisher('/proc_detection/bounding_box', DetectionArray, queue_size=1)
        self.detection_graph, self.score, self.expand = self.load_frozen_model()
        thread.start_new_thread(self.detection, ())
        self.finish_init = True
        rospy.spin()

    ####################################################################################################################
    # This part is highly inspired on https://github.com/GustavZ/realtime_object_detection/blob/r1.0/object_detection.py
    # Licence using MIT licence
    # Copyright to https://github.com/GustavZ
    def load_frozen_model(self):
        if(self.model != self.prev_model):
            self.prev_model = self.model
            rospy.loginfo("load a new frozen model {}".format(self.model))
            detection_graph = tf.Graph()
            try:
                trt_graph = tf.GraphDef()
                with tf.gfile.GFile(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.model + "/trt.pb"), "rb") as f:
                    serialized_trt_graph = f.read()
                trt_graph.ParseFromString(serialized_trt_graph)
                rospy.loginfo("loading graph from file")
            except:
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.model + "/frozen_inference_graph.pb"), 'rb') as fid:
                    serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)

                trt_graph = trt.create_inference_graph(
                input_graph_def=od_graph_def,
                outputs=["detection_boxes:0",
                        "detection_scores:0",
                        "detection_classes:0",
                        "num_detections:0"],
                max_batch_size=1,
                max_workspace_size_bytes=1<<25,
                precision_mode="FP32",
                is_dynamic_op=False,
                minimum_segment_size=50)

                rospy.loginfo("loading graph from scratch")

                with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.model + "/trt.pb"), "wb") as f:
                    f.write(trt_graph.SerializeToString())

            with detection_graph.as_default():

                rospy.loginfo("finish generating tensorrt engine")
                tf.import_graph_def(trt_graph, name='')

                rospy.loginfo("model is loaded!")
            return detection_graph, None, None

        else:

            rospy.loginfo("keep the previous model")
            return self.detection_graph, None, None


    def load_labelmap(self):
        rospy.loginfo('Loading labelmap from label_map.pbtxt')
        label_map = label_map_util.load_labelmap(self.label_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    @staticmethod
    def _node_name(n):
        if n.startswith("^"):
            return n[1:]
        else:
            return n.split(":")[0]

    def init_detection(self):
        rospy.loginfo("Building Graph fpr object detection")
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        with self.detection_graph.as_default():
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            # Define Input and Ouput tensors
            rospy.loginfo("detection graph context")
            try:
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                rospy.loginfo("image_tensor: {}".format(self.image_tensor))
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                rospy.loginfo("detection_boxes: {}".format(self.detection_boxes))
                self.detection_scores = self.detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                rospy.loginfo("detection_scores: {}".format(self.detection_scores))
                self.detection_classes = self.detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                rospy.loginfo("detection_classes: {}".format(self.detection_classes))
                self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                self.fps = FPS2(self.fps_interval).start()

            except:
                rospy.logwarn("Unexpected error: {}".format(sys.exc_info()[0]))



    def image_msg_callback(self, img):
        self.frame = self.cv_bridge.compressed_imgmsg_to_cv2(img, desired_encoding="bgr8")
        if self.frame is None:
            rospy.logwarn("frame is None!")

    def stop(self):
        # End everything
        self.fps.stop()

    def boxes_above_threshold(self, detection_threshold, category_index, classes, scores, boxes):
        pass

    def get_config(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = os.path.join(dir_path, 'config.yml')
        if (file):
            with open(file, 'r') as ymlfile:
                cfg = yaml.load(ymlfile)
        else:
            file = os.path.join(dir_path, 'config.sample.yml')
            with open(file, 'r') as ymlfile:
                cfg = yaml.load(ymlfile)

        self.fps_interval = cfg['fps_interval']
        self.allow_memory_growth = cfg['allow_memory_growth']
        self.det_interval = cfg['det_interval']
        self.det_th = cfg['det_th']
        self.model = cfg['initial_model']

    def detection(self):
        while 1:
            if self.frame is not None:
                with self.detection_graph.as_default():
                    start = datetime.now()

                    # actual Detection
                    # read video frame, expand dimensions and convert to rgb
                    image = self.frame
                    self.frame = None
                    image.setflags(write=1)
                    image_expanded = np.expand_dims(image, axis=0)
                    self.boxes, self.scores, self.classes, num = self.sess.run(
                        [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                        feed_dict={self.image_tensor: image_expanded})

                    bounding_box = self._extract_bounding_box(image.shape[1], image.shape[0])
                    self.bbox_publisher.publish(bounding_box)
                    self.fps.update()
            else:
                time.sleep(1)
    ####################################################################################################################
    @staticmethod
    def _normalize_bbox(box, img_width, img_height):
        top = int(box[0] * img_height)
        left = int(box[1] * img_width)
        bottom = int(box[2] * img_height)
        right = int(box[3] * img_width)
        return [left, right, top, bottom]

    def _extract_bounding_box(self, img_width, img_height):
        list_detection = DetectionArray()
        detections = []
        boxes = self.boxes[0]
        for i in range(boxes.shape[0]):
            if self.scores is not None and self.scores[0][i] > self.detection_thresh:
                detection = Detection()
                detection.bbox = self.create_bounding_box_from_box(boxes[i], img_width, img_height)
                detection.confidence = self.scores[0][i]
                detection.class_name.data = str(self.category_index[self.classes[0][i]]['name'])
                detections.append(detection)
        list_detection.detected_object = list(detections)
        return list_detection


    @classmethod
    def create_bounding_box_from_box(cls, box, img_width, img_height):
        bbox = BoundingBox2D()
        center = Pose2D()
        left, right, top, bottom = cls._normalize_bbox(box, img_width, img_height)
        size_x = right - left
        size_y = bottom - top
        center.x = left + int(size_x / 2)
        center.y = top + int(size_y / 2)
        bbox.center = center
        bbox.size_x = size_x
        bbox.size_y = size_y
        return bbox

    def handle_change_network(self, req):
        # wait until the initialisation model is finished
        while not self.finish_init:
            time.sleep(5)

        json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config/model_path.json')
        with open(json_path) as f:
            models = json.load(f)
        task_name = req.task
        tmp_model = self._get_task_model(task_name, models)

        if tmp_model is not None:
            tmp_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_model)

            if self.image_subscriber is not None:
                self.image_subscriber.unregister()

            if(tmp_model != self.model_path):
                rospy.loginfo('take a tmp_model {}'.format(tmp_model))
                self.model_path = tmp_model
                self.label_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self._get_task_label(task_name, models))


                conf = self._get_model_config(task_name)
                self.topic_subscriber = conf["image_subscriber"]
                self.detection_thresh = conf["detection_thresh"]
                self.num_classes = conf["num_class"]


                self.detection_graph, self.score, self.expand = self.load_frozen_model()
                self.load_labelmap()
                self.init_detection()

                time.sleep(3)

                self.image_subscriber = rospy.Subscriber(self.topic_subscriber, SensorImage, self.image_msg_callback)
            else:
                rospy.loginfo('take the default model')
                conf = self._get_model_config(task_name)
                self.topic_subscriber = conf["image_subscriber"]
                self.detection_thresh = conf["detection_thresh"]
                if self.model_path is not None:
                    self.image_subscriber = rospy.Subscriber(self.topic_subscriber, SensorImage, self.image_msg_callback)
                else:
                    rospy.logwarn('no model found')

        return ChangeNetworkResponse(True)

    def _get_task_model(self, name, models):
        self.model = models.get(name)
        if self.model is not None:
            model_path = self.model + "/frozen_inference_graph.pb"
        else:
            model_path = self.model
        return model_path

    def _get_task_label(self, name, models):
        label_path = models[name] + "/label_map.pbtxt"
        return label_path

    def _get_model_config(self, name):
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config', name + '.json')
        with open(config_path) as f:
            print(f)
            configs = json.load(f)
        return configs


if __name__ == '__main__':
    ObjectDetection()
