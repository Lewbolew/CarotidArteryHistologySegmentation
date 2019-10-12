import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from data_loaders.histology_simple_loader import HistologyData
from models.shape_net import ShapeUNet
from models.unet import UNet

from tqdm import tqdm
import time
import numpy as np
import os
import cv2
import skimage.io as io
import numpy as np
from metrics_evaluator import PerformanceMetricsEvaluator
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torchvision import transforms

import warnings


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

def train_epoch(model, train_dataloader,optimizer):
    model.train()

    log_softmax = nn.LogSoftmax(dim=1)
    criterion_nlloss = nn.NLLLoss()

    print("Training Unet: ")
    average_loss = 0.0
    for imgs, masks in tqdm(train_dataloader):

        imgs, masks = imgs.to(device), masks.to(device)

        # forward pass
        logits = model(imgs)
        # compute loss
        log_logits = log_softmax(logits)

        loss = criterion_nlloss(log_logits, masks)
        average_loss+=loss
        # zero gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()
        optimizer.step()
    average_loss /= len(train_dataloader)
    print("Unet training loss: {:f}".format(average_loss))

def train_network_on_top_of_other(model, train_loader, val_loader, optimizer, 
                                  unet, unet_data_loader, unet_optim, num_epochs, 
                                  path_to_save_best_weights):

    model.train()

    log_softmax = nn.LogSoftmax(dim=1)# Use for NLLLoss()
    softmax = nn.Softmax(dim=1)

    to_tensor = transforms.ToTensor()

    metrics_evaluator = PerformanceMetricsEvaluator()

    criterion_nlloss = nn.NLLLoss()
    criterion_mseloss = nn.MSELoss(size_average=False)
    writer = SummaryWriter('runs/high_pretrained_shape_net/')

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
                if epoch % 4 == 0:
                    for i in range(10):
                        train_epoch(unet, unet_data_loader,unet_optim)
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

                mask_to_encode = masks.numpy()
                mask_to_encode = (np.arange(15) == mask_to_encode[...,None]).astype(float)
                mask_to_encode = torch.from_numpy(np.moveaxis(mask_to_encode, 3, 1)).float().to(device)
                imgs = imgs.to(device)
                masks = masks.to(device)
                mask_to_encode = mask_to_encode.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                unet_output = unet(imgs).detach()
                softmax_unet_output = softmax(unet_output).detach()

                logits, encoded_shape = model(softmax_unet_output)
                _, encoded_mask = model(mask_to_encode)

                log_softmax_logits = log_softmax(logits)
                softmax_logits = softmax(logits)
                first_term = criterion_mseloss(softmax_logits, softmax_unet_output)
                second_term = criterion_mseloss(encoded_shape, encoded_mask)
                lambda_1 = 0.5
                loss = first_term + lambda_1*second_term
                print("First term: ", first_term, "Second term: ", lambda_1*second_term)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                collapsed_softmax_logits = np.argmax(softmax_logits.detach(), axis=1)
                collapsed_softmax_unet = np.argmax(softmax_unet_output.detach(), axis=1)
                for i in range(1):
                    rgb_prediction = collapsed_softmax_logits[i].repeat(3, 1, 1).float()
                    rgb_prediction = np.moveaxis(rgb_prediction.numpy(), 0, -1)
                    converted_img = img_to_visible(rgb_prediction)
                    converted_img = to_tensor(converted_img)

                    rgb_unet_prediction = collapsed_softmax_unet[i].repeat(3, 1, 1).float()
                    rgb_unet_prediction = np.moveaxis(rgb_unet_prediction.numpy(), 0, -1)
                    converted_img_unet = img_to_visible(rgb_unet_prediction)
                    converted_img_unet = to_tensor(converted_img_unet)

                    masks_changed = masks[i].detach().cpu()
                    masks_changed = masks_changed.repeat(3,1,1).float()
                    masks_changed = np.moveaxis(masks_changed.numpy(), 0, -1)
                    masks_changed = img_to_visible(masks_changed)
                    masks_changed = to_tensor(masks_changed)
                    # changed_imgs = torch.cat([imgs[i],imgs[i],imgs[i]]).detach().cpu()

                    third_tensor = torch.cat((imgs[i].cpu(), masks_changed, converted_img_unet, converted_img), -1)
                    if phase == 'val':
                        writer.add_image('ValidationEpoch: {}'.format(str(epoch)), 
                            third_tensor, epoch)
                    else:
                        writer.add_image('TrainingEpoch: {}'.format(str(epoch)), 
                            third_tensor, epoch)

                # statistics
                running_loss += loss.detach().item()
                batch_IoU = 0.0
                for k in range(len(imgs)):
                    batch_IoU += metrics_evaluator.mean_IU(collapsed_softmax_logits.numpy()[k], masks.cpu().numpy()[k])
                batch_IoU /= len(imgs)
                running_IoU+=batch_IoU

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
        os.path.join(path_to_save_best_weights, 'pretrained_shape_net{:2f}.pth'.format(best_val_loss)))

    # Show the timing and final statistics
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_val_loss)) # TODO add IoU




# Choose free GPU
device = torch.device("cuda:{}".format(str(get_freer_gpu())))

ROOT_DIR = 'data/dataset'

# Create Data Loaders
partition = 'train'
shape_train = HistologyData(ROOT_DIR, partition, True)
train_loader = torch.utils.data.DataLoader(shape_train,
                                             batch_size=1, 
                                             shuffle=True,
                                            )
partition = 'val'
shape_val = HistologyData(ROOT_DIR, partition, False)
val_loader = torch.utils.data.DataLoader(shape_val,
                                        batch_size=1,
                                        shuffle=False
                                        )

partition = 'train'
unet_train = HistologyData(ROOT_DIR, partition, True)
unet_loader = torch.utils.data.DataLoader(unet_train,
                                             batch_size=1, 
                                             shuffle=True,
                                            )


# Create model
model = ShapeUNet((15,512,512))
unet = UNet((3,512,512))
model.to(device)
unet.to(device)

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
lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)
NUM_OF_EPOCHS = 40

lr1 = 1e-4
unet_optim = Adam(unet.parameters(), lr=lr1)

train_network_on_top_of_other(model, train_loader, val_loader, optimizer, 
                              unet, unet_loader, unet_optim, NUM_OF_EPOCHS, 
                              'weights/')
