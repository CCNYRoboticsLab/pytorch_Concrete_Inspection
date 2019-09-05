# -*- coding: utf-8 -*-
# dataset.py

# Copyright (c) 2018, Eric Liang Yang @ chiyangliang@gmail.com
# Produced at the Robotics Laboratory of the City College of New York
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
from __future__ import print_function, division
import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import numpy as np
import pickle
import pandas as pd
import time
import random
import cv2



basic_path = '/home/eric/disk/fcnForSpallingCrack/crack_spalling_detect/SPallData/deepLearningBridgeInspection/'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            #img = img.convert('RGB')
            #print(img.shape)
            #if img.size[0]> 200:
            img = img.resize((224, 224))
            return img

def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video):

    if subset == 'training':
        fp=open('SPallData/train_list.txt')
        imglist=fp.readlines()

        fp1=open('SPallData/train_list_crack_2.txt')
        imglist=imglist + fp1.readlines()
        #fp1=open('SPallData/train_list_crack_1.txt')
        #imglist=fp1.readlines()

    elif subset == 'validation':

       fp=open('SPallData/test_list.txt')
       imglist=fp.readlines()

       fp1=open('SPallData/test_list_crack_2.txt')
       imglist=imglist + fp1.readlines()
       #fp1=open('SPallData/test_list_crack_1.txt')
       #imglist= fp1.readlines()


    return imglist


class my_3DST_data(data.Dataset):
    """
    Args:
        root (string): Root directory path.

        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1):
        self.data = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image_third, image_ego, target) where target is class_index of the target class.
        """
        img_path, label   = self.data[index].split(',')
        if os.path.exists(basic_path + img_path):
           #img = np.array(pil_loader(basic_path + img_path))
           img = cv2.imread(basic_path + img_path)
           #print(basic_path + img_path)
           img = cv2.resize(img,(224, 224), interpolation = cv2.INTER_CUBIC)
           img = np.array(img)
        else:
           print(basic_path + img_path)

        sour_img = torch.Tensor(torch.from_numpy(img).float().div(255))
        sour_img = sour_img.permute(2, 0, 1)


        target     = int(label)
        return sour_img, target


    def __len__(self):
        return len(self.data)
