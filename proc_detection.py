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
import yaml
import rospy
import time
import _thread

import numpy as np
import tensorflow as tf

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage as SensorImage
from std_msgs.msg import Empty
from sonia_common.msg import DetectionArray, Detection, ChangeNetworkMsg
from datetime import datetime
from object_detection.utils import label_map_util
from stuff.helper import FPS2
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

        if(tensorrtEnabled):
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

        self.network_subscriber = rospy.Subscriber("/proc_detection/change_network", ChangeNetworkMsg, self.handle_change_network)
        self.network_publisher = rospy.Publisher("/proc_detection/status_ML", ChangeNetworkMsg, queue_size=10)
        self.stop_subscriber = rospy.Subscriber("/proc_detection/stop_topic", Empty, self.stop_topic)
        self.bbox_publisher = rospy.Publisher('/proc_detection/bounding_box', DetectionArray, queue_size=10)
        self.detection_mutex.acquire()
        self.detection_graph = self.load_frozen_model(self.initial_model)
        self.detection_mutex.release()
        _thread.start_new_thread(self.detection, ())
    
    def image_msg_callback(self, img):
        self.frame = self.cv_bridge.compressed_imgmsg_to_cv2(img, desired_encoding="bgr8")
    
    def load_graph_def(self, filepath):
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(filepath, "rb") as f:
            serialized_graph = f.read()
        graph_def.ParseFromString(serialized_graph)
        return graph_def
    
    def load_frozen_model(self, model_name):
        if(model_name != self.prev_model and model_name != "@default"):
            self.prev_model = model_name
            rospy.loginfo("load a new frozen model {}".format(model_name))
            model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "external", 'models' , model_name, 'saved_model')
            detection_function = tf.function()
            if tensorrtEnabled and self.run_with_tensorrt:
                output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "external", 'models' , model_name, 'trt')

                if not os.path.exists(output_dir):
                    params = trt.DEFAULT_TRT_CONVERSION_PARAMS
                    params = params._replace(precision_mode = self.trt_precision_mode)
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
                            inp1 = np.random.normal(size=(1, 1, self.trt_image_width, self.trt_image_height, 3)).astype(matrix_type)
                            yield inp1

                        converter.convert(calibration_input_fn=calibration_input_fn)
                    else:
                        converter.convert()

                    def input_fn():
                        inp1 = np.random.normal(size=(1, 1, self.trt_image_width, self.trt_image_height, 3)).astype(matrix_type)
                        yield inp1

                    converter.build(input_fn=input_fn)
                    converter.save(output_dir)

                loaded_model = tf.saved_model.load(output_dir, tags=['serve'])
            else:
                loaded_model = tf.saved_model.load(model_dir)
            
            # load label names
            label_map = label_map_util.load_labelmap(os.path.join(os.path.dirname(os.path.realpath(__file__)), "external", 'models', model_name, 'labelmap.pbtxt'))
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.num_classes, use_display_name=True)
            self.category_index = label_map_util.create_category_index(categories)
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
            self.prev_model = data.network_name
            self.detection_mutex.acquire()
            self.detection_graph = self.load_frozen_model(data.network_name)
            self.detection_mutex.release()

        self.image_subscriber = rospy.Subscriber(data.topic, SensorImage, self.image_msg_callback)
        self.threshold = data.threshold/100.0
        self.fps = FPS2(self.fps_interval).start()
    
    # function find at: https://github.com/tensorflow/models/blob/75b016b437ab21cbd19dd44451257989fdbb38d6/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb
    def run_inference_for_single_image(self, model, image):
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        model_fn = model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() 
                        for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
            
        return output_dict
    
    def detection(self):
        while not rospy.is_shutdown():
            if self.frame is not None:
                start = datetime.now()

                image = self.frame
                self.frame = None
                image.setflags(write=1)
                image_expanded = np.expand_dims(image, axis=0)
                
                self.detection_mutex.acquire()
                output_dict = self.run_inference_for_single_image(self.detection_graph, image)
                self.detection_mutex.release()

                list_detection = DetectionArray()

                for i in range(output_dict["detection_boxes"].shape[0]):
                    if output_dict["detection_scores"] is not None and output_dict["detection_scores"][i] > self.threshold:
                        detection = Detection()
                        detection.top = output_dict["detection_boxes"][i][0]
                        detection.left = output_dict["detection_boxes"][i][1]
                        detection.bottom = output_dict["detection_boxes"][i][2]
                        detection.right = output_dict["detection_boxes"][i][3]

                        detection.confidence = output_dict["detection_scores"][i]
                        detection.class_name = str(self.category_index[output_dict["detection_classes"][i]]['name'])

                        list_detection.detected_object.append(detection)
                        
                self.bbox_publisher.publish(list_detection)
                self.fps.update()
            else:
                time.sleep(1)
                rospy.loginfo("FPS: -1")
    
    def get_config(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'external', 'config', 'config.yml'), 'r') as ymlfile:
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


if __name__ == '__main__':
    ObjectDetection()
