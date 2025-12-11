# ------------------------------
# Test Images and Masks converted to Tensors
# ------------------------------

from sklearn.preprocessing import LabelEncoder
import os
import sys
# Add the src/ folder to sys.path
from .. import config_helper
import numpy as np  
import segmentation_models as sm
import cv2
import re
from . import file_utils


# from training.train import TrainingLabelCreator 
# This did not work because of line 27 in train.py which uses relative import from scripts ( in line import confighelper)



class ImagetoTensor:
    """Creates training labels by mapping RGB mask values to class indices."""
    def __init__(self,version, continue_training = False, tl_version = '001', use_train_data = False):
        self.color_mapping = {
            (0, 0, 0): 1,  # BG
            (236, 85, 157): 2,  # Pap
            (73, 0, 106): 3,  # Epi
            (248, 123, 168): 4,  # Ker
            (127, 255, 255): 3, (145, 1, 122): 2, (108, 0, 115): 2,
            (255, 127, 127): 3, (181, 9, 130): 2, (216, 47, 148): 3,
            (254, 246, 242): 2, (127, 255, 142): 2
        }
        PROJECT_ROOT = os.path.abspath(os.getcwd())
        base_path = os.path.join(PROJECT_ROOT, "config", "base_config.yaml")
        train_path = os.path.join(PROJECT_ROOT, "config", "training_config.yaml")
        self.config = config_helper.ConfigLoader.load_config(base_path, train_path)
        self.model_type = self.config['training_params']['model_type']
        if continue_training:
            if use_train_data:
                self.test_images_path = os.path.join(PROJECT_ROOT,"data/"+version+"/continue_training1/training_"+tl_version+"/images/train")
                self.test_masks_path = os.path.join(PROJECT_ROOT, "data/"+version+"/continue_training1/training_"+tl_version+"/masks/train")
            else:
                self.test_images_path = os.path.join(PROJECT_ROOT,"data/"+version+"/continue_training1/training_"+tl_version+"/images/test")
                self.test_masks_path = os.path.join(PROJECT_ROOT, "data/"+version+"/continue_training1/training_"+tl_version+"/masks/test")
        else:
            self.test_images_path = os.path.join(PROJECT_ROOT,"data/"+version+"/images/test")
            self.test_masks_path = os.path.join(PROJECT_ROOT, "data/"+version+"/masks/test")


    def create_training_labels(self, mask):
        labels = np.full(mask.shape[:2], 1, dtype=np.uint8)  # default BG
        for color, label in self.color_mapping.items():
            labels[np.all(mask == np.array(color), axis=-1)] = label
        return labels



    def preprocess_data(self, images, masks):
        labelencoder = LabelEncoder()
        n, h, w = masks.shape
        masks_encoded = labelencoder.fit_transform(masks.reshape(-1, 1)).reshape(n, h, w)
        masks_encoded = masks_encoded.astype(np.int32)  # <-- add this

        #masks_encoded = np.expand_dims(masks_encoded, axis=-1)
        #masks_cat = to_categorical(masks_encoded, num_classes=self.n_classes)
        X_images = sm.get_preprocessing(self.model_type)(images)
        return X_images, masks_encoded

    def load_images_and_masks(self):
        images_dir = self.test_images_path
        masks_dir = self.test_masks_path
        image_files = file_utils.natural_sort(os.listdir(images_dir))
        mask_files = file_utils.natural_sort(os.listdir(masks_dir))
        print(image_files)
        images, masks = [], []

        for img_file, mask_file in zip(image_files, mask_files):
            img = cv2.imread(os.path.join(images_dir, img_file), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_size = self.config['training_params']['img_size']
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

            mask = cv2.imread(os.path.join(masks_dir, mask_file), cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            mask = self.create_training_labels(mask)

            images.append(img)
            masks.append(mask)
        return np.array(images), np.array(masks)
    

    




