import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import transforms
import numpy as np
import os
import random
import cv2 as cv
from skimage import io

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    ShiftScaleRotate,
    RandomBrightness,
    RGBShift,
    Blur
)

random.seed(42)

class HistologyData(nn.Module):

    def __init__(self, root_dir, partition, augment):
        self.root_dir = root_dir
        self.list_IDs = os.listdir(os.path.join(self.root_dir, 'y_{}'.format(partition)))
        self.partition = partition
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.augmentator = Compose([
                    # Non destructive transformations
                        VerticalFlip(p=0.6),
                        HorizontalFlip(p=0.6),
                        RandomRotate90(),
                        Transpose(p=0.6),
                        ShiftScaleRotate(p=0.45, scale_limit=(0.1, 0.3)),

                    #     # Non-rigid transformations
                        # ElasticTransform(p=0.25, alpha=160, sigma=180 * 0.05, alpha_affine=120 * 0.03),

                        Blur(blur_limit=3, p=0.2),

                    #     Color augmentation
                        RandomBrightness(p=0.5),
                        RandomContrast(p=0.5),
                        RGBShift(p=0.3),
                        RandomGamma(p=0.5),
                        CLAHE(p=0.5)

                        ]
                    )
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        img_path = os.path.join(self.root_dir, 'x_{}'.format(self.partition), self.list_IDs[index])
        mask_path = os.path.join(self.root_dir, 'y_{}'.format(self.partition), self.list_IDs[index])

        X = io.imread(img_path)
        y = io.imread(mask_path)
        
        if self.augment: 
            augmented = self.augmentator(image=X, mask=y)
            X = augmented['image']
            y = augmented['mask']

        # X = cv.resize(X, (1024,1024), interpolation=cv.INTER_LANCZOS4)
        # y = cv.resize(y, (1024,1024), interpolation=cv.INTER_NEAREST)
        

        X = cv.resize(X, (512,512), interpolation=cv.INTER_LANCZOS4)
        y = cv.resize(y, (512,512), interpolation=cv.INTER_NEAREST)

        X = self.to_tensor(X)
        y = torch.from_numpy(y[:,:, 0]).long()
        return X, y
