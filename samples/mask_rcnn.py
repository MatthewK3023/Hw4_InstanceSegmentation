#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import warnings 
warnings.filterwarnings("ignore")

import tensorflow as tf


# In[4]:


from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[7]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt


# Root directory of the project: /home/dan/VRDL/hw4/Mask_RCNN/samples
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# In[8]:


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# In[9]:


class SubCocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "subcoco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20
    
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

config = SubCocoConfig()
config.display()


# In[12]:


from coco import coco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

dataset_train = coco.CocoDataset()
dataset_train.load_coco(ROOT_DIR+"/samples/dataset", subset="train", year="2014")
dataset_train.prepare()

# dataset_test = coco.CocoDataset()
# dataset_test.load_coco(ROOT_DIR+"/samples/dataset", subset="test", year="2014")
# dataset_test.prepare()

print(dataset_train.class_names)
# print(dataset_test.class_names)


# In[14]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[15]:


# Which weights to start with?
init_with = "imagenet"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

print(model)


# In[ ]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
import datetime
print('start time: ', datetime.datetime.now())
model.train(dataset_train, dataset_train, 
            learning_rate=config.LEARNING_RATE, 
            epochs=100, 
            layers='heads')


# In[16]:


class InferenceConfig(SubCocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# In[17]:


# try predict training set image
image_id = random.choice(dataset_train.image_ids)
print(image_id)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset_train, inference_config, 
                           image_id, use_mini_mask=False)
type(original_image)
# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, figsize=(8, 8))


# In[18]:


# try predict training set image
results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_train.class_names, r['scores'], ax=get_ax())


# In[22]:


import skimage.io
import glob
test_link = glob.glob(ROOT_DIR+"/samples/dataset/test2014/*.jpg")
image = skimage.io.imread(test_link[3])
original_shape = image.shape
image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)


# In[23]:


# try predict testing set image
results = model.detect([image], verbose=1)

r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_train.class_names, r['scores'], ax=get_ax())


# In[33]:


import numpy as np
from itertools import groupby
from pycocotools import mask as maskutil

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle


# In[40]:


cocoGt = COCO('dataset/test.json')
coco_dt = []

for imgid in cocoGt.imgs:
    image = cv2.imread('dataset/test2014/'+cocoGt.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1]
    
    
    results = model.detect([image], verbose=1)

    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_train.class_names, r['scores'], ax=get_ax())
    n_instances = len(r['class_ids'])
    
    if n_instances >0:
        for i in range(n_instances):
            pred = {}
            pred['image_id'] = imgid
            pred['category_id'] = int(r['class_ids'][i])
            pred['segmentation'] = binary_mask_to_rle(r['masks'][:,:,i])
            pred['score'] = float(r['scores'][i])
            coco_dt.append(pred)
print(coco_dt)


# In[41]:


import json
result_json = json.dumps(coco_dt)
with open('0856702.json', 'w') as f:
    json.dump(coco_dt, f)


# In[ ]:




