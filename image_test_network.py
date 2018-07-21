import cv2
import os
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/ssd_mobilenet_v11_coco',
                          'frozen_inference_graph.pb')
images_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images/')
label_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'inference/ssd_mobilenet_v11_coco',
                          'label_map.pbtxt')
num_classes = 10


def get_images_list(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


if __name__ == "__main__":

    images = get_images_list(images_dir)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

        for image_name in images:
            image = cv2.imread(images_dir + image_name)

            image_np = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np})
            image_np = np.squeeze(image_np, axis=0)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            cv2.imshow('object_detection', image)
            # Exit Option
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
