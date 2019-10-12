import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from data_loaders.histology_simple_loader import HistologyData

from models.unet import UNet

from tqdm import tqdm
import time
import os
import cv2
import skimage.io as io
import numpy as np
from scripts.metrics_evaluator import PerformanceMetricsEvaluator
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    return np.argmax(memory_available)

def img_to_visible(img):
    visible_mask = np.zeros((img.shape[0], img.shape[1], 3), 'uint8')
    image = np.zeros((400,400,3), dtype="uint8")
    for i in range(len(mask_values)):
        visible_mask[np.where((img==mask_values[i]).all(axis=-1))] = real_colors[i]
    return visible_mask

def train(model, train_loader, val_loader, optimizer, num_epochs, path_to_save_best_weights):
    model.train()

    log_softmax = nn.LogSoftmax(dim=1)# Use for NLLLoss()
    softmax = nn.Softmax(dim=1)

    # weights = [1.0,1.0,1.0,1.0,1.0, 0.0]
    # class_weights = torch.FloatTensor(weights).to(device)
    criterion_nlloss = nn.NLLLoss()#(weight=class_weights)
    metrics_evaluator = PerformanceMetricsEvaluator()

    to_tensor = transforms.ToTensor()

    writer = SummaryWriter('runs/unet')

    since = time.time()

    best_model_weights = model.state_dict()
    best_IoU = 0.0 
    best_val_loss = 1000000000

    curr_val_loss = 0.0
    curr_training_loss = 0.0
    curr_training_IoU = 0.0
    curr_val_IoU = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:

            if phase == 'train':
                # scheduler.step(best_val_loss)
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_IoU = 0 

            # Iterate over data.
            ind = 0
            for imgs, masks in tqdm(data_loader):
                imgs = imgs.to(device)
                masks = masks.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                logits = model(imgs)
                log_softmax_logits = log_softmax(logits)
                loss = criterion_nlloss(log_softmax_logits, masks)

                                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #


                unet_softmax_collupsed = softmax(logits)
                unet_softmax_collupsed = np.argmax(unet_softmax_collupsed.detach().cpu(), axis=1)

                if ind % 10==0:
                    if phase == 'val':
                        img_name = 'ValidationEpoch: {}'.format(str(epoch))
                    else:
                        img_name = 'TrainingEpoch: {}'.format(str(epoch))

                    rgb_prediction = unet_softmax_collupsed.repeat(3, 1, 1).float()
                    rgb_prediction = np.moveaxis(rgb_prediction.numpy(), 0, -1)
                    converted_img = img_to_visible(rgb_prediction)
                    converted_img = torch.unsqueeze(to_tensor(converted_img), 0)
                    # converted_img = np.moveaxis(converted_img, -1, 0)
                    masks_changed = masks.detach().cpu()
                    masks_changed = masks_changed.repeat(3,1,1).float()
                    masks_changed = np.moveaxis(masks_changed.numpy(), 0, -1)
                    masks_changed = img_to_visible(masks_changed)
                    masks_changed = torch.unsqueeze(to_tensor(masks_changed), 0)

                    # print(np.unique(converted_img, return_counts=True))
                    third_tensor = torch.cat((converted_img, imgs.detach().cpu(), masks_changed), -1)
                    writer.add_image(img_name, 
                            # vutils.make_grid([
                            # imgs.detach().cpu(),
                            # rgb_prediction
                            third_tensor,
                            # ]), 
                    epoch)

                # statistics
                running_loss += loss.detach().item()
                running_IoU += metrics_evaluator.mean_IU(unet_softmax_collupsed.numpy()[0], masks.cpu().numpy()[0])
                ind+=1
            epoch_loss = running_loss / len(data_loader)
            epoch_IoU = running_IoU / len(data_loader)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_IoU))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_val_loss: # TODO add IoU
                best_val_loss = epoch_loss
                best_IoU = epoch_IoU
                best_model_weights = model.state_dict()
    
            if phase == 'val':
                # print(optimizer.param_groups[0]['lr'])
                curr_val_loss = epoch_loss
                curr_val_IoU = epoch_IoU
            else:
                curr_training_loss = epoch_loss
                curr_training_IoU = epoch_IoU

        writer.add_scalars('TrainValIoU', 
                            {'trainIoU': curr_training_IoU,
                             'validationIoU': curr_val_IoU
                            },
                            epoch
                           )
        writer.add_scalars('TrainValLoss', 
                            {'trainLoss': curr_training_loss,
                             'validationLoss': curr_val_loss
                            },
                            epoch
                           ) 
    # Saving best model
    torch.save(best_model_weights, 
        os.path.join(path_to_save_best_weights, 'unet{:2f}.pth'.format(best_val_loss)))

    # Show the timing and final statistics
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_val_loss)) # TODO add IoU



# Choose free GPU
device = torch.device("cuda:{}".format(str(get_freer_gpu())))

# ROOT_DIR = 'data/training'
ROOT_DIR = 'data/dataset'

# Create Data Loaders
partition = 'train'
histology_train = HistologyData(ROOT_DIR, partition, True)
train_loader = torch.utils.data.DataLoader(histology_train,
                                             batch_size=1, 
                                             shuffle=True,
                                            )
partition = 'val'
histology_val = HistologyData(ROOT_DIR, partition, False)
val_loader = torch.utils.data.DataLoader(histology_val,
                                        batch_size=1,
                                        shuffle=False
                                        )
# Create model

model = UNet((3,512,512))
model.to(device)

# Specify optimizer and criterion
lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)
NUM_OF_EPOCHS = 200

# color_codding_for visualisation
# mask_values = [0,1,2,3,4,5]
# real_colors = [(0,0,0), (255,0,0), (0,255,0), (255,0,127), (255,0,255), (128,130,128)]
mask_values = (
                0,1,2,
                3,4,5,
                6,7,8,
                9,10,11,
                12,13,14
               )
# here not RGB but BGR because of OPENCV. 
real_colors = (
                (0,0,0), (255,0,0), (0,255,0),
                (0,0,255), (85,0,0), (0,170,0),
                (255,0,127), (0,255,255), (0,85,0),
                (255,0,255), (255,85,0), (255,165,0),
                (255,255,0), (128,130,128), (128,190,190)  
               )
#training
train(model, train_loader, val_loader, optimizer, NUM_OF_EPOCHS, 'weights/')
