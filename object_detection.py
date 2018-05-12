#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@author: GustavZ
"""
import numpy as np
import os
import tensorflow as tf
import copy
import yaml
from cv_bridge import CvBridge
import cv2
import tarfile
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2
import rospy
from sensor_msgs.msg import Image as SensorImage


# Protobuf Compilation (once necessary)
#os.system('protoc object_detection/protos/*.proto --python_out=.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from stuff.helper import FPS2, WebcamVideoStream, SessionWorker


import time


class ObjectDetection:
    IMAGE_PUBLISHER = '/deep_detector/object_detection'
    IMAGE_SUBSCRIBER = '/usb_cam/image_raw'

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

        self.get_config()
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', self.model_name, 'frozen_inference_graph.pb')
        self.label_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'inference', self.model_name,'label_map.pbtxt')

        self.detection_graph, self.score, self.expand = self.load_frozen_model()
        self.load_labelmap()

        self.init_detection()

        self.image_publisher = rospy.Publisher(self.IMAGE_PUBLISHER,
                                               SensorImage, queue_size=100)
        self.image_subscriber = rospy.Subscriber(
            self.IMAGE_SUBSCRIBER, SensorImage, self.image_msg_callback)

        time.sleep(1)

        self.detection()

        rospy.spin()

    def load_frozen_model(self):
        print('>>>>>>> Loading frozen model into memory <<<<<<<<')
        if not self.split_model:
            print('>>>>> Not spliting model for inference <<<<<')
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            return detection_graph, None, None
        else:
            print('>>>>> Spliting model for optimized inference <<<<<')
            # load a frozen Model and split it into GPU and CPU graphs
            # Hardcoded for ssd_mobilenet

            ###################################################################
            input_graph = tf.Graph()
            with tf.Session(graph=input_graph):
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
        rospy.loginfo('>>>>>>> Loading labelmap from label_map.pbtxt <<<<<<<<')
        label_map = label_map_util.load_labelmap(self.label_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def _node_name(slef, n):
        if n.startswith("^"):
            return n[1:]
        else:
            return n.split(":")[0]

    def init_detection(self):
        rospy.loginfo(">>>>> Building Graph fpr object detection <<<<<")
        # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=self.log_device)
        config.gpu_options.allow_growth = self.allow_memory_growth
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph, config=config) as self.sess:
                # Define Input and Ouput tensors
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                if self.split_model:
                    score_out = self.detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                    expand_out = self.detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                    self.score_in = self.detection_graph.get_tensor_by_name(
                        'Postprocessor/convert_scores_1:0')
                    self.expand_in = self.detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
                    # Threading
                    self.gpu_worker = SessionWorker("GPU", self.detection_graph, config)
                    self.cpu_worker = SessionWorker("CPU", self.detection_graph, config)
                    self.gpu_opts = [score_out, expand_out]
                    self.cpu_opts = [self.detection_boxes, self.detection_scores,
                                self.detection_classes, self.num_detections]
                    gpu_counter = 0
                    cpu_counter = 0
                # Start Video Stream and FPS calculation
                self.fps = FPS2(self.fps_interval).start()
                rospy.loginfo('> Starting Detection')

    def image_msg_callback(self, img):
        self.frame = self.cv_bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")

    def detect_object(self):
        while not rospy.is_shutdown():
            if self.split_model:
                # split model in seperate gpu and cpu session threads
                if self.gpu_worker.is_sess_empty():
                    # read video frame, expand dimensions and convert to rgb
                    image_expanded = np.expand_dims(self.frame, axis=0)
                    #image_expanded = np.expand_dims(
                    #    cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), axis=0)
                    # put new queue
                    gpu_feeds = {self.image_tensor: image_expanded}
                    if self.visualize:
                        gpu_extras = self.frame  # for visualization frame
                    else:
                        gpu_extras = None
                    self.gpu_worker.put_sess_queue(self.gpu_opts, gpu_feeds, gpu_extras)

                g = self.gpu_worker.get_result_queue()
                if g is None:
                    # gpu thread has no output queue. ok skip, let's check cpu thread.
                    pass
                else:
                    # gpu thread has output queue.
                    self.score, self.expand, self.frame = g["results"][0], g["results"][1], g["extras"]

                    if self.cpu_worker.is_sess_empty():
                        # When cpu thread has no next queue, put new queue.
                        # else, drop gpu queue.
                        cpu_feeds = {self.score_in: self.score, self.expand_in: self.expand}
                        cpu_extras = self.frame
                        self.cpu_worker.put_sess_queue(self.cpu_opts, cpu_feeds, cpu_extras)

                c = self.cpu_worker.get_result_queue()
                if c is None:
                    # cpu thread has no output queue. ok, nothing to do. continue
                    time.sleep(0.005)
                    continue
                else:
                    cpu_counter = 0
                    self.boxes, self.scores, self.classes, num, self.frame = c["results"][0], c[
                        "results"][1], c["results"][2], c["results"][3], c["extras"]
            else:
                # default session
                image_expanded = np.expand_dims(
                    cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), axis=0)
                self.boxes, self.scores, self.classes, num = self.sess.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: image_expanded})

            # Visualization of the results of a detection.
            if self.visualize:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    self.frame,
                    np.squeeze(self.boxes),
                    np.squeeze(self.classes).astype(np.int32),
                    np.squeeze(self.scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                if self.vis_text:
                    cv2.putText(self.frame, "fps: {}".format(self.fps.fps_local()), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                cv2.imshow('object_detection', self.frame)

            else:
                self.cur_frames += 1
                # Exit after max frames if no visualization
                for box, score, _class in zip(np.squeeze(self.boxes), np.squeeze(self.scores),
                                              np.squeeze(self.classes)):
                    if self.cur_frames % self.det_interval == 0 and score > self.det_th:
                        label = self.category_index[_class]['name']
                        print("> label: {}\nscore: {}\nbox: {}".format(label, score, box))
                if self.cur_frames >= self.max_frames:
                    pass
            self.fps.update()

    def stop(self):
        # End everything
        if self.split_model:
            self.gpu_worker.stop()
            self.cpu_worker.stop()
        self.fps.stop()

    def boxes_above_threshold(self, detection_threshold, category_index, classes, scores, boxes):
        pass

    def get_config(self):
        if (os.path.isfile('config.yml')):
            with open("config.yml", 'r') as ymlfile:
                cfg = yaml.load(ymlfile)
        else:
            with open("config.sample.yml", 'r') as ymlfile:
                cfg = yaml.load(ymlfile)

        self.video_input = cfg['video_input']
        self.visualize = cfg['visualize']
        self.vis_text = cfg['vis_text']
        self.max_frames = cfg['max_frames']
        self.width = cfg['width']
        self.height = cfg['height']
        self.fps_interval = cfg['fps_interval']
        self.allow_memory_growth = cfg['allow_memory_growth']
        self.det_interval = cfg['det_interval']
        self.det_th = cfg['det_th']
        self.model_name = cfg['model_name']
        self.model_path = cfg['model_path']
        self.label_path = cfg['label_path']
        self.num_classes = cfg['num_classes']
        self.split_model = cfg['split_model']
        self.log_device = cfg['log_device']
        self.ssd_shape = cfg['ssd_shape']

    def detection(self):
        print("> Building Graph")
        # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=self.log_device)
        config.gpu_options.allow_growth = self.allow_memory_growth
        cur_frames = 0
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph, config=config) as sess:
                # Define Input and Ouput tensors
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                if self.split_model:
                    score_out = self.detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                    expand_out = self.detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                    score_in = self.detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                    expand_in = self.detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
                    # Threading
                    gpu_worker = SessionWorker("GPU", self.detection_graph, config)
                    cpu_worker = SessionWorker("CPU", self.detection_graph, config)
                    gpu_opts = [score_out, expand_out]
                    cpu_opts = [detection_boxes, detection_scores, detection_classes, num_detections]
                    gpu_counter = 0
                    cpu_counter = 0
                # Start Video Stream and FPS calculation
                fps = FPS2(self.fps_interval).start()
                #video_stream = WebcamVideoStream(self.video_input, self.width, self.height).start()
                cur_frames = 0
                print("> Press 'q' to Exit")
                print('> Starting Detection')
                while not rospy.is_shutdown():
                    # actual Detection
                    if self.split_model:
                        # split model in seperate gpu and cpu session threads
                        if gpu_worker.is_sess_empty():
                            # read video frame, expand dimensions and convert to rgb
                            image = self.frame
                            image.setflags(write=1)
                            image_expanded = np.expand_dims(image, axis=0)
                            # put new queue
                            gpu_feeds = {image_tensor: image_expanded}
                            if self.visualize:
                                gpu_extras = image  # for visualization frame
                            else:
                                gpu_extras = None
                            gpu_worker.put_sess_queue(self.gpu_opts, gpu_feeds, gpu_extras)

                        g = gpu_worker.get_result_queue()
                        if g is None:
                            # gpu thread has no output queue. ok skip, let's check cpu thread.
                            pass
                        else:
                            # gpu thread has output queue.
                            gpu_counter = 0
                            score, expand, image = g["results"][0], g["results"][1], g["extras"]

                            if cpu_worker.is_sess_empty():
                                # When cpu thread has no next queue, put new queue.
                                # else, drop gpu queue.
                                cpu_feeds = {self.score_in: score, self.expand_in: expand}
                                cpu_extras = image
                                cpu_worker.put_sess_queue(self.cpu_opts, cpu_feeds, cpu_extras)

                        c = cpu_worker.get_result_queue()
                        if c is None:
                            # cpu thread has no output queue. ok, nothing to do. continue
                            time.sleep(0.005)
                            continue  # If CPU RESULT has not been set yet, no fps update
                        else:
                            cpu_counter = 0
                            boxes, scores, classes, num, image = c["results"][0], c["results"][1], c["results"][2], \
                                                                 c["results"][3], c["extras"]
                    else:
                        # default session
                        image = self.frame
                        image.setflags(write=1)
                        image_expanded = np.expand_dims(image, axis=0)
                        boxes, scores, classes, num = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: image_expanded})


                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    image_message = self.cv_bridge.cv2_to_imgmsg(image, encoding="bgr8")
                    self.image_publisher.publish(image_message)
                    fps.update()


        # End everything
        if self.split_model:
            gpu_worker.stop()
            cpu_worker.stop()
        fps.stop()

if __name__ == '__main__':
    ObjectDetection()
