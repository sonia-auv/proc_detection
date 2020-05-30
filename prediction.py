# Two things to run this script
# 1. run "pip install graphqlclient"
# 2. Fill in <API-KEY-HERE>

# This script will create an end to end example on Labelbox account with predictions. This includes...
# - A project called "Predictions Example"
# - A new dataset "Predictions Example: Ollie Dataset"
# - Three datarows in that dataset
# - A new prediction model
# - A prediction for each datarow in the Ollie dataset


import json
from graphqlclient import GraphQLClient

import sys
import os
import tensorflow as tf
import numpy as np
import time
from object_detection.utils import label_map_util
import cv2
import csv
from StringIO import StringIO
from cv_bridge import CvBridge
from PIL import Image
import yaml

client = None

def me():
    res_str = client.execute("""
    query GetUserInformation {
      user {
        id
        organization{
          id
        }
      }
    }
    """)

    res = json.loads(res_str)
    return res['data']['user']


def createDataset(name):
    res_str = client.execute("""
    mutation CreateDatasetFromAPI($name: String!) {
      createDataset(data:{
        name: $name
      }){
        id
      }
    }
    """, {'name': name})

    res = json.loads(res_str)
    return res['data']['createDataset']['id']


def createProject(name):
    res_str = client.execute("""
    mutation CreateProjectFromAPI($name: String!) {
      createProject(data:{
        name: $name
      }){
        id
      }
    }
    """, {'name': name})

    res = json.loads(res_str)
    return res['data']['createProject']['id']


def completeSetupOfProject(project_id, dataset_id, labeling_frontend_id):
    res_str = client.execute("""
    mutation CompleteSetupOfProject($projectId: ID!, $datasetId: ID!, $labelingFrontendId: ID!){
      updateProject(
        where:{
          id:$projectId
        },
        data:{
          setupComplete: "2018-11-29T20:46:59.521Z",
          datasets:{
            connect:{
              id:$datasetId
            }
          },
          labelingFrontend:{
            connect:{
              id:$labelingFrontendId
            }
          }
        }
      ){
        id
      }
    }
    """, {
        'projectId': project_id,
        'datasetId': dataset_id,
        'labelingFrontendId': labeling_frontend_id
    })

    res = json.loads(res_str)
    return res['data']['updateProject']['id']


def configure_interface_for_project(ontology, project_id, interface_id, organization_id):
    res_str = client.execute("""
      mutation ConfigureInterfaceFromAPI($projectId: ID!, $customizationOptions: String!, $labelingFrontendId: ID!, $organizationId: ID!) {
        createLabelingFrontendOptions(data:{
          customizationOptions: $customizationOptions,
          project:{
            connect:{
              id: $projectId
            }
          }
          labelingFrontend:{
            connect:{
              id:$labelingFrontendId
            }
          }
          organization:{
            connect:{
              id: $organizationId
            }
          }
        }){
          id
        }
      }
    """, {
        'projectId': project_id,
        'customizationOptions': json.dumps(ontology),
        'labelingFrontendId': interface_id,
        'organizationId': organization_id,
    })

    res = json.loads(res_str)
    return res['data']['createLabelingFrontendOptions']['id']


def get_image_labeling_interface_id():
    res_str = client.execute("""
      query GetImageLabelingInterfaceId {
        labelingFrontends(where:{
          iframeUrlPath:"https://image-segmentation-v4.labelbox.com"
        }){
          id
        }
      }
    """)

    res = json.loads(res_str)
    return res['data']['labelingFrontends'][0]['id']


def create_prediction_model(name, version):
    res_str = client.execute("""
      mutation CreatePredictionModelFromAPI($name: String!, $version: Int!) {
        createPredictionModel(data:{
          name: $name,
          version: $version
        }){
          id
        }
      }
    """, {
      'name': name,
      'version': version
    })

    res = json.loads(res_str)
    return res['data']['createPredictionModel']['id']

def attach_prediction_model_to_project(prediction_model_id, project_id):
    res_str = client.execute("""
      mutation AttachPredictionModel($predictionModelId: ID!, $projectId: ID!){
        updateProject(where:{
          id: $projectId
        }, data:{
          activePredictionModel:{
            connect:{
              id: $predictionModelId
            }
          }
        }){
          id
        }
      }
    """, {
      'predictionModelId': prediction_model_id,
      'projectId': project_id
    })

    res = json.loads(res_str)
    return res['data']['updateProject']['id']


def create_prediction(label, prediction_model_id, project_id, data_row_id):
    res_str = client.execute("""
      mutation CreatePredictionFromAPI($label: String!, $predictionModelId: ID!, $projectId: ID!, $dataRowId: ID!) {
        createPrediction(data:{
          label: $label,
          predictionModelId: $predictionModelId,
          projectId: $projectId,
          dataRowId: $dataRowId,
        }){
          id
        }
      }
    """, {
        'label': label,
        'predictionModelId': prediction_model_id,
        'projectId': project_id,
        'dataRowId': data_row_id
    })

    res = json.loads(res_str)
    return res['data']['createPrediction']['id']


def create_datarow(row_data, external_id,dataset_id):
    res_str = client.execute("""
      mutation CreateDataRowFromAPI(
        $rowData: String!,
        $externalId: String,
        $datasetId: ID!
      ) {
        createDataRow(data:{
          externalId: $externalId,
          rowData: $rowData,
          dataset:{
            connect:{
              id: $datasetId
            }
          }
        }){
          id
        }
      }
    """, {
        'rowData': row_data,
        'externalId': external_id,
        'datasetId': dataset_id
    })

    res = json.loads(res_str)
    return res['data']['createDataRow']['id']

def _normalize_bbox(box, img_width, img_height):
  top = int(box[0] * img_height)
  left = int(box[1] * img_width)
  bottom = int(box[2] * img_height)
  right = int(box[3] * img_width)
  return [left, right, top, bottom]

  def _extract_bounding_box(boxes, scores, img_width, img_height):
    list_detection = DetectionArray()
    detections = []
    boxes = boxes[0]
    for i in range(boxes.shape[0]):
      if scores is not None and scores[0][i] > self.detection_thresh:
        detection = Detection()
        detection.bbox = self.create_bounding_box_from_box(boxes[i], img_width, img_height)
        detection.confidence = self.scores[0][i]
        detection.class_name.data = str(self.category_index[self.classes[0][i]]['name'])
        detections.append(detection)
    list_detection.detected_object = list(detections)
    return list_detection


if __name__ == "__main__":

  # setup config file
  dir_path = os.path.dirname(os.path.realpath(__file__))
  file = os.path.join(dir_path, 'config.yml')
  if (file):
    with open(file, 'r') as ymlfile:
      cfg = yaml.load(ymlfile)
  else:
    file = os.path.join(dir_path, 'config.sample.yml')
    with open(file, 'r') as ymlfile:
      cfg = yaml.load(ymlfile)

  detection_thresh = cfg['pred_thresh']
  num_classes = cfg['pred_classes']
  csv_path = cfg['pred_csv_path']
  images_path = cfg['pred_image_path']
  dataset_name = cfg['pred_dataset_name']
  project_name = cfg['pred_project_name']
  client = GraphQLClient('https://api.labelbox.com/graphql')
  client.inject_token('Bearer ' + cfg['pred_api_key'])
  model_name = cfg['pred_model']
  classes_filter = cfg['pred_classes_filter']
  begin = cfg['pred_begin']
  end = cfg['pred_end']


  cv_bridge = CvBridge()

  label_map = label_map_util.load_labelmap("models/" + model_name + "/label_map.pbtxt")
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)


  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile("models/" + model_name + "/frozen_inference_graph.pb", 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=detection_graph, config=config) as sess:
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
      detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      user_info = me()
      org_id = user_info['organization']['id']
      project_id = createProject(project_name)
      print('Created Project: %s' % (project_id))
      dataset_id = createDataset(dataset_name)
      print('Created Dataset: %s' % (dataset_id))
      interface_id = get_image_labeling_interface_id()

      # configure tools
      ontology = {
          "tools": [
              {
                  "color": "Red",
                  "tool": "rectangle",
                  "name": "bat"
              },
              {
                  "color": "Blue",
                  "tool": "rectangle",
                  "name": "wolf"
              },
              {
                  "color": "Green",
                  "tool": "rectangle",
                  "name": "vetalas"
              },
              {
                  "color": "Yellow",
                  "tool": "rectangle",
                  "name": "jiangshi"
              },
              {
                  "color": "Magenta",
                  "tool": "rectangle",
                  "name": "vampire"
              },
              {
                  "color": "Pink",
                  "tool": "rectangle",
                  "name": "draugr"
              },
              {
                  "color": "Cornsilk",
                  "tool": "rectangle",
                  "name": "answag"
              }
          ]
      }

      configure_interface_for_project(ontology, project_id, interface_id, org_id)
      completeSetupOfProject(project_id, dataset_id, interface_id)
      print('Attached Dataset and Interface to Created Project')

      prediction_model_id = create_prediction_model('test', 1)
      attach_prediction_model_to_project(prediction_model_id, project_id)

      print('Created and attached prediction model: %s' % (prediction_model_id))

      fh = open(csv_path)
      fh.readline()

      inc = 0

      while inc < begin:
        fh.readline()
        inc += 1

      for line in fh:
        line = line.rstrip()
        print("{} : {}".format(inc, line))

        url_list = line.split("/")
        image_input_fqn = os.path.join(images_path, '{image_dir}/{filename}'.format(image_dir=url_list[-2], filename=url_list[-1]))

        image = cv2.imread(image_input_fqn)
        image.setflags(write=1)
        image_expanded = np.expand_dims(image, axis=0)
        boxes, scores, classes, num = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_expanded})
        boxes = boxes[0]

        prediction_dict = dict(list())

        for i in range(boxes.shape[0]):
          if scores is not None and scores[0][i] > detection_thresh:
            if str(category_index[classes[0][i]]['name']) in classes_filter:
              top = int(boxes[i][0] * image.shape[0])
              left = int(boxes[i][1] * image.shape[1])
              bottom = int(boxes[i][2] * image.shape[0])
              right = int(boxes[i][3] * image.shape[1])
              if(prediction_dict.has_key(str(category_index[classes[0][i]]['name']))):
                prediction_dict[str(category_index[classes[0][i]]['name'])].append({"geometry": [{"x": left, "y": top}, {"x": right, "y": top}, {"x": right, "y": bottom}, {"x": left, "y": bottom}]})
              else:
                prediction_dict[str(category_index[classes[0][i]]['name'])] = [{"geometry": [{"x": left, "y": top}, {"x": right, "y": top}, {"x": right, "y": bottom}, {"x": left, "y": bottom}]}]

        # in case of an http 503
        while True:
          try:
            data_row_id = create_datarow(line, url_list[-1], dataset_id)
            prediction_id = create_prediction(json.dumps(prediction_dict), prediction_model_id, project_id, data_row_id)
          except:
            continue
          break

        if(inc >= end and end != -1) :
          break
        inc += 1

  print('Go to https://app.labelbox.com/projects/%s/overview and click start labeling' % (project_id))
