import os
import io
import ast
import json

import boto3
import torch
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.autograd import Variable

# PyTorch models
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from resnext import resnext50

# Custom PyTorch conv net builder based on fast.ai code
from conv_builder import ConvnetBuilder

import logging

# set the constants for the content types
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

# get variables from environment
IMG_SIZE = int(os.environ.get('IMAGE_SIZE', '224'))
MODEL_ARCH = os.environ.get('MODEL_ARCH', 'resnext50')
MODEL_FILE_NAME = os.environ.get('MODEL_FILE_NAME', 'resnext50.h5')
CLASSES_FILE_NAME = os.environ.get('CLASSES_FILE_NAME', 'classes.json')

logger = logging.getLogger(__name__)

preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(IMG_SIZE),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocess the image
def preprocess_image(img):
    logger.info("Preprocessing image")
    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)
    return img_tensor

# Create the PyTorch model class
def create_model(classes):
    logger.info('Creating model with architecture: {}'.format(MODEL_ARCH))
    arch=globals()[MODEL_ARCH]
    builder = ConvnetBuilder(arch, len(classes))
    return builder.model

# Create the Resnet Conv Model and load saved weights
def model_fn(model_dir):
    logger.info('Loading the model.')
    model_info = {}
    
    # load the classes dir
    classes_file = os.path.join(model_dir, CLASSES_FILE_NAME)
    logger.info('Classes file is: {}'.format(classes_file))
    with open(classes_file) as f:
        model_info['classes'] = ast.literal_eval(json.load(f))
    logger.info("Number of classes is: {}".format(len(model_info['classes'])))
    
    # load the model
    model_file = os.path.join(model_dir, MODEL_FILE_NAME)
    logger.info('Model file is: {}'.format(model_file))
    
    logger.info('Loading PyTorch model')
    model = create_model(model_info['classes'])
    state_dict = torch.load(model_file, map_location=lambda storage, loc:storage)
    model.load_state_dict(state_dict)
    model.eval()
    model_info['model'] = model
    return model_info
        
# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    if content_type == JPEG_CONTENT_TYPE:
        logger.info('Processing jpeg image.')
        img_pil = Image.open(io.BytesIO(request_body))
        img_tensor = preprocess(img_pil)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        logger.info("Returning image as PyTorch Variable.")
        return img_variable
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    log_preds = model['model'](input_object).data.numpy()
    
    logger.info("Getting best prediction")
    preds = np.argmax(np.exp(log_preds), axis=1)
    
    logger.info("Getting class and confidence score")
    classes = model['classes']
    response = {}
    response['class'] = classes[preds.item()]
    response['confidence'] = np.exp(log_preds[:,preds.item()]).item()
    logger.info(response)
    return response

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))    