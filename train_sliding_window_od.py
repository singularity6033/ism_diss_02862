# import the necessary packages
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from sklearn.preprocessing import LabelBinarizer
from utils.change_cnn_input_size import change_input_size
from utils.dataloader import ip_sw_dataset
from utils.utils import get_classes
import tensorflow as tf
from utils import config
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import os

matplotlib.use('Agg')
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# initialize the initial learning rate, number of epochs to train for and batch size
init_lr = 1e-4
num_epochs = 10
batch_size = 2
save_path = './logs_sl/1'
model_path = './vgg_cifar100/cifar100vgg_scratch.json'
weights_path = './vgg_cifar100/cifar100vgg_scratch.h5'
classes_path = os.path.join(config.ORIG_BASE_PATH, 'coco_classes.txt')
input_shape = [300, 300]

class_names, num_classes = get_classes(classes_path)
train_annotation_path = os.path.join(config.ORIG_BASE_PATH, 'shuffled_train_set.txt')
val_annotation_path = os.path.join(config.ORIG_BASE_PATH, 'shuffled_val_set.txt')

lb = LabelBinarizer()
lb.fit(list(class_names))

# load train and val set
with open(train_annotation_path) as f:
    train_lines = f.readlines()
with open(val_annotation_path) as f:
    val_lines = f.readlines()
num_train = len(train_lines)
num_val = len(val_lines)
epoch_step_train = num_train // batch_size
epoch_step_val = num_val // batch_size

# initialize both the training and testing image generators
gen_train = ip_sw_dataset(train_lines, input_shape, lb, batch_size, aug=True, scale=1.5,
                                   win_step=16, roi_size=(64, 64), mini_size=(32, 32)).generate()
gen_val = ip_sw_dataset(val_lines, input_shape, lb, batch_size, aug=False, scale=1.5,
                                 win_step=16, roi_size=(64, 64), mini_size=(32, 32)).generate()

# load pre-trained VGG CNN (cifar100) and drop off the head FC layer
# and change the input size to (224, 224, 3)
with open(model_path, 'r') as file:
    model_json = file.read()
origin_model = model_from_json(model_json)
origin_model.load_weights(weights_path)
base_model = Model(inputs=origin_model.input, outputs=origin_model.layers[-8].output)
new_model = change_input_size(base_model, 300, 300, 3)

# construct the head of the model that will be placed on top of the the base model
headModel = new_model.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5, name='dropout_10')(headModel)
headModel = Dense(num_classes, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=new_model.input, outputs=headModel)
# model.summary()
# loop over all layers in the base_model and freeze them so they will
# not be updated during the first training process
for layer in base_model.layers:
    layer.trainable = False
# compile model
opt = Adam(lr=init_lr)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# fine-tuning the head of the network
print("[INFO] training head...")
H = model.fit(
    gen_train,
    steps_per_epoch=epoch_step_train,
    validation_data=gen_val,
    validation_steps=epoch_step_val,
    epochs=10)

# save records of loss and accuracy
with open(os.path.join(save_path, "loss.txt"), 'a') as f:
    f.write(str(H.history["loss"]))
    f.write("\n")
with open(os.path.join(save_path, "val_loss.txt"), 'a') as f:
    f.write(str(H.history["val_loss"]))
    f.write("\n")
with open(os.path.join(save_path, "accuracy.txt"), 'a') as f:
    f.write(str(H.history["accuracy"]))
    f.write("\n")
with open(os.path.join(save_path, "val_accuracy.txt"), 'a') as f:
    f.write(str(H.history["val_accuracy"]))
    f.write("\n")

# plot the training loss and accuracy
print("[INFO] plotting results...")
N = num_epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"])
plt.plot(np.arange(0, N), H.history["val_loss"])
plt.legend(['train', 'test'])
plt.title('loss')
plt.savefig(os.path.sep.join([save_path, "sliding_window_od_loss.png"]), dpi=300, format="png")
plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"])
plt.plot(np.arange(0, N), H.history["val_accuracy"])
plt.legend(['train', 'test'])
plt.title('accuracy')
plt.savefig(os.path.sep.join([save_path, "sliding_window_od_accuracy.png"]), dpi=300, format="png")

# Save model to disk
print("Save model weights to disk")
model.save_weights(os.path.sep.join([save_path, "sliding_window_od.h5"]))
model_json = model.to_json()
with open(os.path.sep.join([save_path, "sliding_window_od.json"]), "w") as json_file:
    json_file.write(model_json)
