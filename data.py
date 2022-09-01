import os
import io
import PIL
import numpy as np
from glob import glob
import numpy as  np
import sys
import random
from tqdm import tqdm
import copy

#Check folder function, if an required folder is not present, then it creates one
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

# Normalizes the image
def normalize(images):
    return (images.astype(np.float32)/255.0)

# Retains the image to color format
def denormalize(images):
    return np.clip(images*255.0, a_min=0.001, a_max=255.0).astype(np.uint8)

# An iterator to read, process and feed images to the network
class Dataset_Dispenser():
    def __init__(self, data_path, jpeg_quality, patch_size, batch_size):
        self.data_path = data_path
        self.jpeg_quality = jpeg_quality
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.images_input, self.images_label = self.load_images()

    def load_images(self):
        #pre-load images. For example, using the train dataset, 800 images will be preloaded on the memory
        file_list = glob(os.path.join(self.data_path,"*.*"))
        input_list = []
        label_list = []
        for f in tqdm(file_list):
            # read image
            label = PIL.Image.open(f).convert('RGB')

            # compress
            buffer = io.BytesIO()
            label.save(buffer, format='jpeg', quality=self.jpeg_quality)
            input = PIL.Image.open(buffer)

            # normalization and appending
            input_list.append(normalize(np.array(input)))
            label_list.append(normalize(np.array(label)))

        return input_list, label_list

    def __iter__(self):
        return self

    def __next__(self):
        patches_input = []
        patches_label = []
        for i in range(self.batch_size):
            rand_idx = random.randint(0,len(self.images_label)-1)
            crop_y = random.randint(0, self.images_label[rand_idx].shape[0] - self.patch_size-1)
            crop_x = random.randint(0, self.images_label[rand_idx].shape[1] - self.patch_size-1)
            input_patch = self.images_input[rand_idx][crop_y:crop_y+self.patch_size, crop_x:crop_x+self.patch_size]
            label_patch = self.images_label[rand_idx][crop_y:crop_y+self.patch_size, crop_x:crop_x+self.patch_size]
            patches_input.append(input_patch)
            patches_label.append(label_patch)

        patches_input = np.array(patches_input)
        patches_label = np.array(patches_label)
        return patches_input, patches_label
