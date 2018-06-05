import os, json, traceback
import urllib.parse

import ast
import json

import boto3
import torch
import dill as dill 
import numpy as np

import utils

_MODEL_PATH = os.path.join('/opt/ml/', 'model')  # Path where all your model(s) live in
_TMP_IMG_PATH = os.path.join('/tmp', 'images')
_TMP_IMG_FILE = os.path.join(_TMP_IMG_PATH, 'image.jpg')

IMG_SIZE = int(os.environ.get('IMAGE_SIZE', '224'))

class ClassificationService:
    class __ClassificationService:
        
        def __init__(self, model_path):
            self.model_path = model_path
            self._model = None
            self._classes = None
        
        def __str__(self):
            return repr(self) + self.model_path
            
        @property
        def model(self):
            if not self._model:
                # Get the model filename
                model_file = utils.get_file_with_ext(self.model_path, '.pt')
                print(f'Model file is: {model_file}')
                
                self._model = torch.load(model_file, map_location='cpu', pickle_module=dill).cpu()
                print("Created model successfully")
            return self._model
    
        @property
        def classes(self):
            if not self._classes:
                classes_file = utils.get_file_with_ext(self.model_path, '.json')
                print(f'Classes file is: {classes_file}')

                with open(classes_file) as f:
                    self._classes = ast.literal_eval(json.load(f))
            return self._classes
    
    instance = None
    
    def __init__(self, model_path):
        if not ClassificationService.instance:
            ClassificationService.instance = ClassificationService.__ClassificationService(model_path)
        else:
            ClassificationService.instance.model_path = model_path
            ClassificationService.instance._model = None
            ClassificationService.instance._classes = None
    
    def __getattr__(self, name):
        return getattr(self.instance, name)
    
    @property
    def model(self):
        return self.instance.model
        
    @property
    def classes(self):
        return self.instance.classes

print("Creating predictor object")
predictor = ClassificationService(_MODEL_PATH)

def predict(img_bytes):
    """
    Prediction given the request input
    :param json_input: [dict], request input
    :return: [dict], prediction
    """
    
    print("Got new request")
    utils.write_test_image(img_bytes, _TMP_IMG_PATH, _TMP_IMG_FILE)
    
    print("Opening test image")
    test_img = utils.open_image(_TMP_IMG_FILE)
    print("Pre-processing test image")
    p_img = utils.preproc_img(test_img, IMG_SIZE)
    
    print("Calling model")
    log_preds = predictor.model(p_img).data.numpy()
    
    print("Getting best prediction")
    preds = np.argmax(np.exp(log_preds), axis=1)
    
    print("Getting class and confidence score")
    classes = predictor.classes
    pred_class = classes[preds.item()]
    confidence = np.exp(log_preds[:,preds.item()]).item()
    
    print(f'Returning class: {pred_class} and confidence score: {confidence}')
    return { 'class': pred_class, 'confidence': confidence }

