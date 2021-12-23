import os
import random

import cv2
import numpy as np
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# img1 = np.ones((224, 224, 3))
# img2 = np.ones((224, 224, 3))
# x = [img1, img2]
# y = np.array(x)

# x = ['a', 'b']
# y = [1, 2]
# z = random.choices(x, y)[0]
# print('%s_classes.txt' % 'coco')
from tensorflow.keras.layers import Input

from utils import config

# coco = COCO(config.TRAIN_ANNOTS)
# cats = coco.loadCats(coco.getCatIds())
# catIds = [cat['id'] for cat in cats]
# coco_classes = [cat['name'] for cat in cats]
#
# print(str(coco_classes.index('cake')))

# a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# tr, te = train_test_split(a, test_size=0.1)
print(4 + True)
s = (None, None, 3)
inputs = Input(shape=s)
x, y, z = (None, None, 3)
# x = inputs.numpy()
