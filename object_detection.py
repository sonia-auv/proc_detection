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
import yaml
import json
import rospy
import time
import _thread

import numpy as np
import tensorflow as tf

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage as SensorImage
from sonia_common.msg import DetectionArray, Detection, BoundingBox2D
from std_srvs.srv import Trigger
from datetime import datetime
from sonia_common.srv import ChangeNetwork, ChangeNetworkRequest, ChangeNetworkResponse
from object_detection.utils import label_map_util
from stuff.helper import FPS2, SessionWorker
from threading import Lock

tensorrtEnabled = False
try:
    import tensorflow.contrib.tensorrt as trt
    tensorrtEnabled = True
except:
    rospy.loginfo("cannot import tensorrt, probably because the gpu is not detected!")


class ObjectDetection:
    def __init__(self):
        rospy.init_node('proc_detection')
        rospy.loginfo("found gpu: {}".format(tf.test.gpu_device_name()))

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
        self.allow_memory_growth = None
        self.trt_precision_mode = None
        self.trt_is_dynamic_op = None
        self.trt_segment_size = None
        self.frame = None

        self.get_config()

        self.network_service = rospy.Service("/proc_detection/change_network", ChangeNetwork, self.handle_change_network)
        self.stop_service = rospy.Service("/proc_detection/stop_topic", Trigger, self.stop_topic)
        self.bbox_publisher = rospy.Publisher('/proc_detection/bounding_box', DetectionArray, queue_size=1)
        self.detection_mutex.acquire()
        self.detection_graph = self.load_frozen_model(self.initial_model)
        self.detection_mutex.release()
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
        if(model_name != self.prev_model):
            self.prev_model = model_name
            rospy.loginfo("load a new frozen model {}".format(self.model))
            detection_graph = tf.Graph()
            if tensorrtEnabled:
                try:
                    trt_graph = self.load_graph_def(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.model + "/trt.pb"))
                    rospy.loginfo("loading graph from file")
                except:
                    od_graph_def = self.load_graph_def(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.model + "/frozen_inference_graph.pb"))
                    trt_graph = trt.create_inference_graph(
                    input_graph_def=od_graph_def,
                    outputs=["detection_boxes:0",
                            "detection_scores:0",
                            "detection_classes:0",
                            "num_detections:0"],
                    max_batch_size=1,
                    max_workspace_size_bytes=1<<25,
                    precision_mode=self.trt_precision_mode,
                    is_dynamic_op=self.trt_is_dynamic_op,
                    minimum_segment_size=self.trt_segment_size)

                    rospy.loginfo("loading graph from scratch")

                    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.model + "/trt.pb"), "wb") as f:
                        f.write(trt_graph.SerializeToString())
                    
                    with detection_graph.as_default():

                        rospy.loginfo("finish generating tensorrt engine")
                        tf.import_graph_def(trt_graph, name='')

                        rospy.loginfo("model is loaded!")
            else:
                graph_def = self.load_graph_def(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.model + "/frozen_inference_graph.pb"))
                with detection_graph.as_default():
                    rospy.loginfo("finish importing the graph file")
                    tf.import_graph_def(graph_def, name='')

                    rospy.loginfo("model is loaded!")
            
            label_map = label_map_util.load_labelmap(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.model + "/labelmap.pbtxt"))
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)
            self.category_index = label_map_util.create_category_index(categories)
            
            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth = self.allow_memory_growth
            with detection_graph.as_default():
                self.sess = tf.Session(graph=detection_graph, config=config)
                # Define Input and Ouput tensors
                rospy.loginfo("detection graph context")
                try:
                    self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    rospy.loginfo("image_tensor: {}".format(self.image_tensor))
                    self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    rospy.loginfo("detection_boxes: {}".format(self.detection_boxes))
                    self.detection_scores = detection_graph.get_tensor_by_name(
                        'detection_scores:0')
                    rospy.loginfo("detection_scores: {}".format(self.detection_scores))
                    self.detection_classes = detection_graph.get_tensor_by_name(
                        'detection_classes:0')
                    rospy.loginfo("detection_classes: {}".format(self.detection_classes))
                    self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                except:
                    rospy.logwarn("Unexpected error: {}".format(sys.exc_info()[0]))
            
            return detection_graph

        else:

            rospy.loginfo("keep the previous model")
            return self.detection_graph
    
    def stop_topic(self):
        if self.image_subscriber is not None:
            self.image_subscriber.unregister()
            self.image_subscriber = None
        
        return Trigger(True)

    def handle_change_network(self, req):
        if self.image_subscriber is not None:
            self.image_subscriber.unregister()
            self.image_subscriber = None
        
        if req.network_name != self.prev_model:
            self.prev_model = req.network_name
            self.detection_mutex.acquire()
            self.detection_graph = self.load_frozen_model(req.network_name)
            self.detection_mutex.release()

        self.image_subscriber = rospy.Subscriber(req.topic, SensorImage, self.image_msg_callback)
        self.threshold = req.threshold
        self.fps = FPS2(self.fps_interval).start()

        return ChangeNetworkResponse(True)
    
    def detection(self):
        while not rospy.is_shutdown():
            if self.frame is not None:
                with self.detection_graph.as_default():
                    start = datetime.now()

                    image = self.frame
                    self.frame = None
                    image.setflags(write=1)
                    image_expanded = np.expand_dims(image, axis=0)
                    
                    self.detection_mutex.acquire()
                    boxes, scores, classes, num = self.sess.run(
                        [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                        feed_dict={self.image_tensor: image_expanded})
                    self.detection_mutex.release()

                    list_detection = DetectionArray()

                    for i in range(boxes[0].shape[0]):
                        if scores is not None and scores[0][i] > self.detection_thresh:
                            detection = Detection()
                            detection.top = boxes[0][i][0]
                            detection.left = boxes[0][i][1]
                            detection.bottom = boxes[0][i][2]
                            detection.right = boxes[0][i][3]

                            detection.confidence = scores[0][i]
                            detection.class_name.data = str(self.category_index[classes[0][i]]['name'])

                            list_detection.detected_object.append(detection)
                            
                    self.bbox_publisher.publish(list_detection)
                    self.fps.update()
            else:
                time.sleep(1)
    
    def get_config(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yml'), 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        
        self.initial_model = cfg['initial_model']
        self.fps_limit = cfg['fps_limit']
        self.allow_memory_growth = cfg['allow_memory_growth']
        self.trt_precision_mode = cfg['trt_precision_mode']
        self.trt_is_dynamic_op = cfg['trt_is_dynamic_op']
        self.trt_segment_size = cfg['trt_segment_size']


if __name__ == '__main__':
    ObjectDetection()
