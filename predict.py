import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

# from shape_net import ShapeUNet
from models.SR_UnetOriginal import SH_UNet
# from SR_2unet import SH_2UNet
from data_loaders.histology_simple_loader import HistologyData

from models.unet import UNet
from models.r2_unet import R2U_Net, R2AttU_Net

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
from torchvision.utils import save_image, make_grid
from torchvision import transforms
# from sklearn.metrics import jaccard_similarity_score
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

# Data Loading
ROOT_DIR = 'data/dataset'
partition = 'test'
shape_test = HistologyData(ROOT_DIR, partition, False)
test_loader = torch.utils.data.DataLoader(shape_test, batch_size=1, shuffle=False)

# Model Creation
device = torch.device("cuda:{}".format(str(get_freer_gpu())))

PATH_TO_THE_WEIGHTS = 'weights/unet0.180934.pth'
model = UNet((3,512,512))
# model = SH_UNet()

# model = R2AttU_Net(img_ch=3, output_ch=15)
# model = SH_2UNet()

model.load_state_dict(torch.load(PATH_TO_THE_WEIGHTS))
model.to(device)
# Evaluation Techniques
metrics_evaluator = PerformanceMetricsEvaluator()

# Evaluation
iou_of_the_model = 0
mean_acc_of_the_model = 0

softmax = nn.Softmax(dim=1)

temporary_counter = 0


# with torch.no_grad():
#     unique_ind = 0
#     to_tensor = transforms.ToTensor()

#     for (test_imgs, test_masks) in tqdm(test_loader):
#         mask_to_encode = (np.arange(15) == test_masks.numpy()[...,None]).astype(float)
#         mask_to_encode = torch.from_numpy(np.moveaxis(mask_to_encode, 3, 1)).float().to(device)

#         test_imgs = test_imgs.to(device)
#         test_masks = test_masks.to(device)
#         mask_to_encode = mask_to_encode.to(device)

#         # unet_prediction = model(test_imgs)
#         # np.save('logits.npy', logits)
#         # temporary_counter+=1
#         # if temporary_counter > 2:
#         #     break
#         unet_prediction, shape_net_encoded_prediction, shape_net_final_prediction = model(test_imgs)

#         softmax_logits = softmax(shape_net_final_prediction)
#         collapsed_softmax_logits = np.argmax(softmax_logits.detach(), axis=1)

#         softmax_logits_unet = softmax(unet_prediction)
#         collapsed_softmax_unet = np.argmax(softmax_logits_unet.detach(), axis=1)


#         for i in range(len(test_imgs)):
#             rgb_prediction = collapsed_softmax_logits[i].repeat(3, 1, 1).float()
#             rgb_prediction = np.moveaxis(rgb_prediction.numpy(), 0, -1)
#             converted_img = img_to_visible(rgb_prediction)
#             converted_img = to_tensor(converted_img)

#             # rgb_unet_prediction = collapsed_softmax_unet[i].repeat(3, 1, 1).float()
#             # rgb_unet_prediction = np.moveaxis(rgb_unet_prediction.numpy(), 0, -1)
#             # converted_img_unet = img_to_visible(rgb_unet_prediction)
#             # converted_img_unet = to_tensor(converted_img_unet)

#             masks_changed = test_masks[i].detach().cpu()
#             masks_changed = masks_changed.repeat(3,1,1).float()
#             masks_changed = np.moveaxis(masks_changed.numpy(), 0, -1)
#             masks_changed = img_to_visible(masks_changed)
#             masks_changed = to_tensor(masks_changed)
#             # changed_imgs = torch.cat([test_imgs[i],test_imgs[i],test_imgs[i]]).detach().cpu()

#             # changed_imgs = make_grid([test_imgs[i].cpu()], normalize=True, range=(0,255))
#             third_tensor = torch.cat((test_imgs[i].cpu(), masks_changed, converted_img), -1)
#             # third_tensor = torch.cat((changed_imgs, masks_changed, converted_img_unet, converted_img), -1)

#             save_image(converted_img, 'predictions/shape_net_with_pretraining/{}'.format(str(unique_ind))+'.png')
#             # save_image(test_imgs[i].cpu(), 'predictions/x/{}'.format(str(unique_ind))+'.png')
#             # save_image(masks_changed, 'predictions/y/{}'.format(str(unique_ind))+'.png')

#             unique_ind+=1





#         batch_iou = 0
#         batch_mean_acc = 0
#         for k in range(len(test_imgs)):
#             batch_iou += metrics_evaluator.mean_IU(collapsed_softmax_logits.numpy()[k], test_masks.cpu().numpy()[k])
#             batch_mean_acc += metrics_evaluator.mean_accuracy(collapsed_softmax_logits.numpy()[k], test_masks.cpu().numpy()[k])
#         batch_iou /= len(test_imgs)
#         batch_mean_acc /= len(test_imgs)
#         mean_acc_of_the_model += batch_mean_acc
#         iou_of_the_model += batch_iou

#     mean_acc_of_the_model /= len(test_loader)
#     iou_of_the_model /= len(test_loader)

#     print('IoU: ', iou_of_the_model)
#     print('Mean Accuracy: ', mean_acc_of_the_model)