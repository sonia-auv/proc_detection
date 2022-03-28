#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@original-author: GustavZ
@author: Club SONIA
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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

try:
    from yaml import CLoader as Loader
except:
    from yaml import Loader

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
            rospy.loginfo("load a new frozen model {}".format(model_name))
            model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "external", model_name)
            detection_function = tf.function()
            if tensorrtEnabled:
                params = trt.DEFAULT_TRT_CONVERSION_PARAMS
                params = params._replace(precision_mode = self.trt_precision_mode)
                converter = trt.TrtGraphConverterV2(input_saved_model=model_dir, conversion_params=params)
                converter.convert()
                #converter.build()
                converter.save(output_graph)

                loaded_model = tf.saved_model.load(output_graph, tags=[tag_constants.SERVING])
                detection_function = loaded_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            else:
                detection_function = tf.saved_model.load(model_dir)
            
            rospy.loginfo("model is loaded")
            
            return detection_function

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
        
        # Handle models with masks:
        # if 'detection_masks' in output_dict:
        #     # Reframe the the bbox mask to the image size.
        #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        #             output_dict['detection_masks'], output_dict['detection_boxes'],
        #             image.shape[0], image.shape[1])      
        #     detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
        #                                     tf.uint8)
        #     output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
            
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
                #boxes, scores, classes, num = self.detection_graph(
                #    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                #    feed_dict={self.image_tensor: image_expanded})
                self.detection_mutex.release()

                #print(output_dict)
                list_detection = DetectionArray()

                for i in range(output_dict["detection_boxes"].shape[0]):
                    if output_dict["detection_scores"] is not None and output_dict["detection_scores"][i] > self.threshold:
                        detection = Detection()
                        detection.top = output_dict["detection_boxes"][i][0]
                        detection.left = output_dict["detection_boxes"][i][1]
                        detection.bottom = output_dict["detection_boxes"][i][2]
                        detection.right = output_dict["detection_boxes"][i][3]

                        detection.confidence = output_dict["detection_scores"][i]
                        #detection.class_name.data = str(self.category_index[output_dict["detection_classes"][i]]['name'])

                        list_detection.detected_object.append(detection)
                        
                self.bbox_publisher.publish(list_detection)
                self.fps.update()
            else:
                time.sleep(1)
    
    def get_config(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'external', 'config', 'config.yml'), 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=Loader)
        
        self.initial_model = cfg['initial_model']
        self.fps_limit = cfg['fps_limit']
        self.allow_memory_growth = cfg['allow_memory_growth']
        self.trt_precision_mode = cfg['trt_precision_mode']
        self.trt_is_dynamic_op = cfg['trt_is_dynamic_op']
        self.trt_segment_size = cfg['trt_segment_size']


if __name__ == '__main__':
    ObjectDetection()
