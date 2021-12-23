from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json

from nets.classifier import get_vgg_classifier
from nets.rpn import get_rpn
from nets.vgg import VGG16
from utils.change_cnn_input_size import change_input_size


def get_model(model_path, weights_path, num_classes, num_anchors=9):
    with open(model_path, 'r') as file:
        model_json = file.read()
    origin_model = model_from_json(model_json)
    origin_model.load_weights(weights_path)
    base_model = Model(inputs=origin_model.input, outputs=origin_model.layers[-10].output)
    main_model = change_input_size(base_model, None, None, 3)
    base_layers = main_model.output
    # inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    # base_layers = VGG16(inputs, model_path, weights_path)
    classifier = get_vgg_classifier(base_layers, roi_input, 7, num_classes)
    # base layer is the shared feature map from previous cnn
    # build region proposal network
    # rpn = get_rpn(base_layers_output, num_anchors)
    rpn = get_rpn(base_layers, num_anchors)
    # build classifier cnn
    # classifier = get_vgg_classifier(base_layers_output, roi_input, 7, num_classes)
    model_rpn = Model(main_model.input, rpn)
    model_all = Model([main_model.input, roi_input], rpn + classifier)
    # model_rpn.summary()
    # model_all.summary()
    return model_rpn, model_all


def get_predict_model(num_classes, num_anchors=9):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None, None, 512))
    base_layers = VGG16(inputs)
    rpn = get_rpn(base_layers, num_anchors)
    classifier = get_vgg_classifier(feature_map_input, roi_input, 7, num_classes)
    model_rpn = Model(inputs, rpn + [base_layers])
    model_classifier_only = Model([feature_map_input, roi_input], classifier)
    return model_rpn, model_classifier_only
