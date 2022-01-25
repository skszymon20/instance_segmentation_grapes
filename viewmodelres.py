from traceback import print_tb
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

from ex import WGISDMaskedDataset
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from utilities import plot_item
from torch.utils.data import DataLoader

NUM_CLASSES = 2
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
#replace head predictor (boxes)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
#replace mask predictor
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
model.load_state_dict(torch.load("./models/model1.pt", map_location=torch.device('cpu')))

BATCH_SIZE = 2
dataset=WGISDMaskedDataset('./wgisd/', source='train')
dataloader = DataLoader(dataset, BATCH_SIZE, collate_fn=lambda s: tuple(zip(*s)))
model.eval()

im, targets = dataset[0]
res = model([im])
res = res[0] # only one image
boxes = res['boxes']
masks = res['masks']
srcboxes = targets['boxes']
srcmasks = targets['masks']
print(targets.keys())
print(boxes.detach().numpy().shape, masks.detach().numpy().shape)
print(srcboxes.numpy().shape, srcmasks.numpy().shape)
plot_item(im.numpy(), boxes.detach().numpy(), masks.detach().numpy(), savename='./test.jpg')