# import the necessary packages
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils.change_cnn_input_size import change_input_size
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from utils.iou import compute_iou
from utils import config
from imutils import paths
import numpy as np
import pickle
import cv2
import os

"""
    generate region proposals for coco train2017 dataset by using selective search method.
    it can be further used to fine-tuning the pre-trained vgg nn classifier to a region-based cnn object detector.
"""
# initialize the x_train and y_train to store the info (image and label) of
# roi (region of interest) generated from coco train2017
x_data = []
y_data = []

# grab all image paths in the coco train2017 images directory
imagePaths = list(paths.list_images(config.TRAIN_IMAGES))

# using cocoAPI to load coco annotation information
coco = COCO(config.TRAIN_ANNOTS)
# load COCO categories and super categories
cats = coco.loadCats(coco.getCatIds())
catIds = [cat['id'] for cat in cats]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # show a progress report
    print("[INFO] processing train image {}/{}...".format(i + 1, len(imagePaths)))
    # extract the filename from the file path and use it to derive
    # the path to the XML annotation file
    filename = imagePath.split(os.path.sep)[-1]
    filename = filename[:filename.rfind(".")]
    img_id = int(filename.lstrip('0'))
    annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    annInfos = coco.loadAnns(annIds)
    # initialize our list of ground-truth bounding boxes
    gtBoxes = []
    # coco bounding box format: (x - top left, y - top left, width, height)
    # loop over all 'object' elements
    for annInfo in annInfos:
        # extract the label and bounding box coordinates
        label_id = annInfo['category_id']
        xMin, yMin, bw, bh = annInfo['bbox']
        xMax = xMin + bw
        yMax = yMin + bh
        # update our list of ground-truth bounding boxes
        gtBoxes.append((xMin, yMin, xMax, yMax, label_id))
    # load the input image from disk
    image = cv2.imread(imagePath)
    # run selective search on the image and initialize our list of
    # proposed boxes
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    proposedRects = []
    # loop over the rectangles generated by selective search
    for (x, y, w, h) in rects:
        # convert our bounding boxes from (x, y, w, h) to (startX, startY, startX, endY)
        proposedRects.append((x, y, x + w, y + h))
    # loop over region proposals
    for proposedRect in proposedRects:
        # unpack the proposed rectangle bounding box
        (propStartX, propStartY, propEndX, propEndY) = proposedRect
        # loop over the ground-truth bounding boxes
        for gtBox in gtBoxes:
            # compute the intersection over union between the two
            # boxes and unpack the ground-truth bounding box
            iou = compute_iou(gtBox[:-1], proposedRect)
            (gtStartX, gtStartY, gtEndX, gtEndY, label_id) = gtBox
            # initialize the ROI and output path
            roi = None
            outputPath = None
            # check to see if the IOU is greater than 80% *and* that
            # we have not hit our positive count limit
            if iou > 0.8:
                # extract the ROI and then derive the output path to the positive instance
                roi = image[propStartY:propEndY, propStartX:propEndX]
                roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                cat_id = catIds.index(label_id)
                x_data.append(roi)
                y_data.append(cat_id)

# convert into numpy format
x_data = np.array(x_data)
y_data = np.array(y_data)
(x_train, x_val, y_train, y_val) = train_test_split(x_data, y_data, test_size=0.20)
#
# print("[INFO] saving x_train...")
# f = open('x_train.pickle', "wb")
# f.write(pickle.dumps(x_train))
#
# print("[INFO] saving y_train...")
# f = open('y_train.pickle', "wb")
# f.write(pickle.dumps(y_train))

"""
    fine-tuning pre-trained vgg nn to rcnn object detector
"""

# initialize the initial learning rate, number of epochs to train for and batch size
INIT_LR = 1e-4
EPOCHS = 10
BS = 32

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# load pre-trained VGG CNN (cifar100) and drop off the head FC layer
# and change the input size to (224, 224, 3)
with open(os.path.sep.join([config.VGG_BASE_PATH, "cifar100vgg_scratch.json"]), 'r') as file:
    model_json = file.read()
origin_model = model_from_json(model_json)
origin_model.load_weights(os.path.sep.join([config.VGG_BASE_PATH, "cifar100vgg_scratch.h5"]))
base_model = Model(inputs=origin_model.input, outputs=origin_model.layers[-8].output)
base_model.summary()
new_model = change_input_size(base_model, 224, 224, 3)
new_model.summary()

# construct the head of the model that will be placed on top of the the base model
headModel = new_model.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5, name='dropout_10')(headModel)
headModel = Dense(80, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=new_model.input, outputs=headModel)
# loop over all layers in the base_model and unfreeze them so they will be updated during the first training process
for layer in base_model.layers:
    layer.trainable = True
# compile model
opt = Adam(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# fine-tuning the head of the network
print("[INFO] training head...")
H = model.fit(
    aug.flow(x_train, y_train, batch_size=BS),
    ssteps_per_epoch=len(x_train) // BS,
    validation_data=(x_val, y_val),
    validation_steps=len(x_val) // BS,
    epochs=EPOCHS)
# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(config.MODEL_PATH, save_format="h5")

# plot the training loss and accuracy
print("[INFO] plotting results...")
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.legend(['train', 'test'])
plt.title('loss')
plt.savefig(os.path.sep.join([config.PLOT_PATH, "rcnn_coco_loss.png"]), dpi=300, format="png")
plt.figure()
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.legend(['train', 'test'])
plt.title('accuracy')
plt.savefig(os.path.sep.join([config.MODEL_PLOT_PATH, "rcnn_coco_accuracy.png"]), dpi=300, format="png")
