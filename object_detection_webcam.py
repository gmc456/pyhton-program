import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import random
import time
import cv2
import json
import time
import socket
import requests
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from datetime import datetime
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


url = 'http://localhost:3002/recognizedObjects'
token = 'qvrrzmysc381y6nff8w0m'    
starttime = time.time()

while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
 
def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)
 
  model_dir = pathlib.Path(model_dir)/"saved_model"
  model = tf.saved_model.load(str(model_dir))
  return model
 
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
#print(category_index)
 
model_name = 'ssd_inception_v2_coco_2017_11_17'
detection_model = load_model(model_name)
 
def run_inference_for_single_image(model, image):
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
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
     
  return output_dict

def show_inference(model, frame):
  #take the frame from webcam feed and convert that to array
  image_np = np.array(frame)
  # Actual detection.
     
  output_dict = run_inference_for_single_image(model, image_np)
 
  # Getting the current date and time
  dt = datetime.now()
  # dt_string = dt.strftime("%d-%m-%Y-%H:%M:%S")
  dt_string = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
  # Getting hostname
  computerName = socket.gethostname();
  # Final JSON to send
  objects = []

  # Loop over all detections and draw detection box if confidence is above minimum threshold
  scores = output_dict['detection_scores']
  classes = output_dict['detection_classes']

  for i in range(len(scores)):
    if ((scores[i] > 0.50) and (scores[i] <= 1.0)):
      # Draw label
      category = category_index[int(classes[i])] # Look up object name from "labels" array using class index
      object_name = category['name']
      objectAndValue = {'object':object_name, 'value': int(scores[i]*100)}
      objects.append(objectAndValue)

  information = {'id_estacion':computerName, 'timestamp':dt_string, 'objectsDetected':objects, 'token' :token}
  inforToString = json.dumps(information)
  #client = connect_mqtt()
  #client.loop_start()
  #publish(client, inforToString)
  x = requests.post(url, json = information)
  print(x.text)


#Now we open the webcam and start detecting objects
video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    print("LECTURA")
    re,frame = video_capture.read()
    #Imagenp=show_inference(detection_model, frame)
    show_inference(detection_model, frame)
    
    #cv2.imshow('object detection', cv2.resize(Imagenp, (800,600)))  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(20.0 - ((time.time() - starttime) % 20.0))

video_capture.release()
cv2.destroyAllWindows()