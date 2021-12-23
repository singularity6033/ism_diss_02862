# import the necessary packages
import os

# define the base path to the output of the vgg_cifar100 model
VGG_BASE_PATH = "./vgg_cifar100"
VGG_PLOT_PATH = "./vgg_cifar100/plot"

# define the base path to the *original* coco dataset and then use
# the base path to derive the coco image and annotations directories
ORIG_BASE_PATH = "./dataset/coco"
TRAIN_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "train2017"])
TRAIN_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations/instances_train2017.json"])
TEST_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "val2017"])
TEST_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations/instances_val2017.json"])

# define the path to the base output directory
BASE_OUTPUT = "rcnn_coco"
# define the path to the output model, plot output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "coco_object_detector_rcnn_VGG16_224.h5"])
ENCODER_PATH = os.path.sep.join([BASE_OUTPUT, "coco_label_encoder.pickle"])
MODEL_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot"])

# define the number of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

# define the maximum number of positive ROI to be generated from each image
MAX_POSITIVE = 20

# initialize the input dimensions to the network
INPUT_DIMS = (224, 224)

# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.99
