#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@author: GustavZ
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
from deep_detector.msg import DetectionArray, Detection, BoundingBox2D
from geometry_msgs.msg import Pose2D
import json
from deep_detector.srv import ChangeNetwork, ChangeNetworkRequest, ChangeNetworkResponse

# Protobuf Compilation (once necessary)
# os.system('protoc object_detection/protos/*.proto --python_out=.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from stuff.helper import FPS2, SessionWorker

import time


class ObjectDetection:
    def __init__(self):
        rospy.init_node('deep_detector')

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

        self.get_config()
        self.model_path = None
        self.label_path = None


        self.network_service = rospy.Service("deep_detector/change_network", ChangeNetwork, self.handle_change_network)
        #self.image_publisher = rospy.Publisher(self.topic_publisher, SensorImage, queue_size=1)
        self.bbox_publisher = rospy.Publisher('/deep_detector/bounding_box', DetectionArray, queue_size=1)

        rospy.spin()

    def load_frozen_model(self):
        rospy.loginfo('Loading frozen model into memory')
        if not self.split_model:
            rospy.loginfo('Not spliting model for inference')
            detection_graph = tf.Graph()
	    with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            return detection_graph, None, None
        else:
            rospy.loginfo('Spliti for optimized inference')
            # load a frozen Model and split it into GPU and CPU graphs
            # Hardcoded for ssd_mobilenet

            ###################################################################
            input_graph = tf.Graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(graph=input_graph,config=config):
                if self.ssd_shape == 600:
                    shape = 7326
                else:
                    shape = 1917
                score = tf.placeholder(tf.float32, shape=(None, shape, self.num_classes),
                                       name="Postprocessor/convert_scores")
                expand = tf.placeholder(tf.float32, shape=(None, shape, 1, 4),
                                        name="Postprocessor/ExpandDims_1")
                for node in input_graph.as_graph_def().node:
                    if node.name == "Postprocessor/convert_scores":
                        score_def = node
                    if node.name == "Postprocessor/ExpandDims_1":
                        expand_def = node
            #####################################################################
            #####################################################################
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    dest_nodes = ['Postprocessor/convert_scores', 'Postprocessor/ExpandDims_1']

                    edges = {}
                    name_to_node_map = {}
                    node_seq = {}
                    seq = 0
                    for node in od_graph_def.node:
                        n = self._node_name(node.name)
                        name_to_node_map[n] = node
                        edges[n] = [self._node_name(x) for x in node.input]
                        node_seq[n] = seq
                        seq += 1
                    for d in dest_nodes:
                        assert d in name_to_node_map, "%s is not in graph" % d

                    nodes_to_keep = set()
                    next_to_visit = dest_nodes[:]

                    while next_to_visit:
                        n = next_to_visit[0]
                        del next_to_visit[0]
                        if n in nodes_to_keep:
                            continue
                        nodes_to_keep.add(n)
                        next_to_visit += edges[n]

                    nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])
                    nodes_to_remove = set()

                    for n in node_seq:
                        if n in nodes_to_keep_list:
                            continue
                        nodes_to_remove.add(n)
                    nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])

                    keep = graph_pb2.GraphDef()
                    for n in nodes_to_keep_list:
                        keep.node.extend([copy.deepcopy(name_to_node_map[n])])

                    remove = graph_pb2.GraphDef()
                    remove.node.extend([score_def])
                    remove.node.extend([expand_def])
                    for n in nodes_to_remove_list:
                        remove.node.extend([copy.deepcopy(name_to_node_map[n])])

                    with tf.device('/gpu:0'):
                        tf.import_graph_def(keep, name='')
                    with tf.device('/cpu:0'):
                        tf.import_graph_def(remove, name='')
            #####################################################################
            return detection_graph, score, expand

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
        # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=self.log_device)
        config.gpu_options.allow_growth = self.allow_memory_growth
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph, config=config) as self.sess:
                # Define Input and Ouput tensors
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                self.detection_scores = self.detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                self.detection_classes = self.detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                if self.split_model:
                    score_out = self.detection_graph.get_tensor_by_name(
                        'Postprocessor/convert_scores:0')
                    expand_out = self.detection_graph.get_tensor_by_name(
                        'Postprocessor/ExpandDims_1:0')
                    self.score_in = self.detection_graph.get_tensor_by_name(
                        'Postprocessor/convert_scores_1:0')
                    self.expand_in = self.detection_graph.get_tensor_by_name(
                        'Postprocessor/ExpandDims_1_1:0')
                    # Threading
                    self.gpu_worker = SessionWorker("GPU", self.detection_graph, config)
                    self.cpu_worker = SessionWorker("CPU", self.detection_graph, config)
                    self.gpu_opts = [score_out, expand_out]
                    self.cpu_opts = [self.detection_boxes, self.detection_scores,
                                     self.detection_classes, self.num_detections]
                # Start Video Stream and FPS calculation
                self.fps = FPS2(self.fps_interval).start()

    def image_msg_callback(self, img):
        self.frame = self.cv_bridge.compressed_imgmsg_to_cv2(img, desired_encoding="bgr8")
        self.detection()

    def stop(self):
        # End everything
        if self.split_model:
            self.gpu_worker.stop()
            self.cpu_worker.stop()
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
        self.split_model = cfg['split_model']
        self.log_device = cfg['log_device']
        self.ssd_shape = cfg['ssd_shape']
        self.visualize = cfg['visualize']

    def detection(self):
        with self.detection_graph.as_default():
            start = datetime.now()

            # actual Detection
            # read video frame, expand dimensions and convert to rgb
            image = self.frame
            if self.split_model:
        # split model in seperate gpu and cpu session threads
                if self.gpu_worker.is_sess_empty():
                    if image is not None:
                        image.setflags(write=1)
                        image_expanded = np.expand_dims(image, axis=0)
                        # put new queue
                        gpu_feeds = {self.image_tensor: image_expanded}
                        if self.visualize:
                            gpu_extras = image
                        else:
                            gpu_extras = None
                        self.gpu_worker.put_sess_queue(self.gpu_opts, gpu_feeds, gpu_extras)
                    else:
                        rospy.logwarn("No image feeded to the network")

                g = self.gpu_worker.get_result_queue()
                if g is None:
                    # gpu thread has no output queue. ok skip, let's check cpu thread.
                    pass
                else:
                    # gpu thread has output queue.
                    gpu_counter = 0
                    score, expand, image = g["results"][0], g["results"][1], g["extras"]

                    if self.cpu_worker.is_sess_empty():
                        # When cpu thread has no next queue, put new queue.
                        # else, drop gpu queue.
                        cpu_feeds = {self.score_in: score, self.expand_in: expand}
                        cpu_extras = image
                        self.cpu_worker.put_sess_queue(self.cpu_opts, cpu_feeds, cpu_extras)

                c = self.cpu_worker.get_result_queue()
                if c is None:
                    # cpu thread has no output queue. ok, nothing to do. continue
                    time.sleep(0.005)
                    return  # If CPU RESULT has not been set yet, no fps update
                else:
                    cpu_counter = 0
                    self.boxes, self.scores, self.classes, num, image = c["results"][0], c["results"][1], \
                        c["results"][2], \
                        c["results"][3], c["extras"]
            else:
                if image is not None:
                    image.setflags(write=1)
                    image_expanded = np.expand_dims(image, axis=0)
                    self.boxes, self.scores, self.classes, num = self.sess.run(
                        [self.detection_boxes, self.detection_scores,
                            self.detection_classes, self.num_detections],
                        feed_dict={self.image_tensor: image_expanded})
                else:
                    rospy.logwarn("No image feeded to the network")

            #vis_util.visualize_boxes_and_labels_on_image_array(
            #    image,
            #     np.squeeze(self.boxes),
            #     np.squeeze(self.classes).astype(np.int32),
            #     np.squeeze(self.scores),
            #     self.category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=8)
            bounding_box = self._extract_bounding_box(image.shape[1], image.shape[0])
            self.bbox_publisher.publish(bounding_box)
            #image_message = self.cv_bridge.cv2_to_imgmsg(image, encoding="bgr8")
            #self.image_publisher.publish(image_message)




            self.fps.update()

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
                conf = self._get_model_config(task_name)
                self.topic_subscriber = conf["image_subscriber"]
                self.detection_thresh = conf["detection_thresh"]
                if self.model_path is not None:
                    self.image_subscriber = rospy.Subscriber(self.topic_subscriber, SensorImage, self.image_msg_callback)

        return ChangeNetworkResponse(True)

    def _get_task_model(self, name, models):
        model = models.get(name)
        if model is not None:
            model_path = model + "/frozen_inference_graph.pb"
        else:
            model_path = model
        return model_path

    def _get_task_label(self, name, models):
        label_path = models[name] + "/label_map.pbtxt"
        return label_path

    def _get_model_config(self, name):
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config', name + '.json')
        with open(config_path) as f:
            configs = json.load(f)
        return configs


if __name__ == '__main__':
    ObjectDetection()
