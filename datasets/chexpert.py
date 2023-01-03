import torch
from torch.utils.data import Dataset
import os
import numpy as np
from scipy.misc import imread
from PIL import Image
import random
from random import sample

'''
 Chexpert:(train:223415 valid:235)
 0:NF 1:EC 2:Cd 3:AO 4:LL 5:Ed 6:Co 7:Pn 8:A 9:Pnx 10:Ef 11:PO 12:Fr 13:SD
'''

class Chexpert(Dataset):
    def __init__(self, path_image, path_list, transform, num_class, target_class=0):
        self.path_list = path_list
        self.transform = transform
        self.path_image = path_image
        self.num_class=num_class

        self.img_list = []
        self.img_label = []
        self.labels = []
        self.protected = []

        self.dict = [{'1.0': 1.0, '': 0.0, '0.0': 0.0, '-1.0': -1.0},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        self.dict_protected = {'Female':1, 'Male':0}

        with open(self.path_list, "r") as fileDescriptor:
            line = fileDescriptor.readline()
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.strip('\n').split(',')
                    imagePath = os.path.join(self.path_image, lineItems[0])
                    imageLabel = lineItems[5:5+14]

                    for idx,_ in enumerate(imageLabel):
                        imageLabel[idx]=self.dict[0][imageLabel[idx]]

                    if lineItems[1] in ['Male','Female']:  # filter some items with gender 'UNKNOWN
                        self.img_list.append(imagePath)
                        self.img_label.append(imageLabel)
                        self.labels.append(imageLabel[target_class])
                        self.protected.append(self.dict_protected[lineItems[1]])


    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # label = torch.zeros((self.num_class),dtype=torch.float)
        # for i in range(0, self.num_class):
        #     label[i] = self.img_label[idx][i]
        label = self.labels[idx]

        return img, label, self.protected[idx]

    def __len__(self):
        return len(self.img_list)
