import os
import random
import json
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
import json

from utils.dataloader import sliding_window_dataset

# x = sliding_window_dataset('', input_shape=(224, 224), batch_size=4, aug=True, scale=1.5, win_step=16,
#                            roi_size=(100, 100)).test(
x = cv2.imread('./dataset/coco/train2017/000000339019.jpg')
# jsonData = '{"a":1}'
# model = load_model('logs/ep005-loss2.512-val_loss3.599.h5')
# model.summary()
# text = json.loads(jsonData)
# for t in text:
#     p = t
#     print(p)
# p['a'] = 2
# print(text)
# a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# for i, aa in enumerate(a):
#     aa = 0
# print(a)
# tr, te = train_test_split(a, test_size=0.1)
# print(4 + True)
# s = (None, None, 3)
# inputs = Input(shape=s)
# x, y, z = (None, None, 3)
# x = inputs.numpy()
