from pickle import TRUE
from random import shuffle
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(0)
#already written Dataset taken from wgisd repo based on directory structures
class WGISDMaskedDataset(Dataset):
    def __init__(self, root, transforms=None, source='train'):
        self.root = root
        self.transforms = transforms
        
        if source not in ('train', 'test'):
            print('source should be by "train" or "test"')
            return None

        source_path = os.path.join(root, f'{source}_masked.txt')
        with open(source_path, 'r') as fp:
          lines = fp.readlines()
          ids = [l.rstrip() for l in lines]# removes /n at the end of each line

        self.imgs = [os.path.join(root, 'data', f'{id}.jpg') for id in ids]
        self.masks = [os.path.join(root, 'data', f'{id}.npz') for id in ids]
        self.boxes = [os.path.join(root, 'data', f'{id}.txt') for id in ids]

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        box_path = self.boxes[idx]

        img = Image.open(img_path).convert("RGB")

        # From TorchVision documentation:
        # 
        # The models expect a list of Tensor[C, H, W], in the range 0-1. 
        # The models internally resize the images so that they have a minimum 
        # size of 800. This option can be changed by passing the option min_size 
        # to the constructor of the models.
        
        if self.transforms is None:
            pass
        else:
            img = np.array(img)
            img = self.transforms(torch.as_tensor(img, dtype=torch.uint8))

        img = np.array(img)
        # Normalize
        img = (img - img.min()) / np.max([img.max() - img.min(), 1])
        # Move the channels axe to the first position, getting C, H, W instead H, W, C
        img = np.moveaxis(img, -1, 0)
        img = torch.as_tensor(img, dtype=torch.float32)  

        # Loading masks:
        #
        # As seen in WGISD (README.md):
        # 
        # After assigning the NumPy array to a variable M, the mask for the 
        # i-th grape cluster can be found in M[:,:,i]. The i-th mask corresponds 
        # to the i-th line in the bounding boxes file.
        #
        # According to Mask RCNN documentation in Torchvision:
        #
        # During training, the model expects both the input tensors, as well as 
        # a targets (list of dictionary), containing:
        # (...) 
        # masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each 
        # instance
        wgisd_masks = np.load(mask_path)['arr_0'].astype(np.uint8)
        masks = np.moveaxis(wgisd_masks, -1, 0) 

        num_objs = masks.shape[0]
        all_text = np.loadtxt(box_path, delimiter = " ", dtype = np.float32)
        wgisd_boxes = all_text[:,1:]
        assert(wgisd_boxes.shape[0] == num_objs)

        # IMPORTANT: Torchvision considers 0 as background. So, let's make grapes
        # grapes as class 1
        labels = np.ones(num_objs, dtype=np.int64)

        # According to WGISD:
        #
        # These text files follows the "YOLO format"
        # 
        # CLASS CX CY W H
        # 
        # class is an integer defining the object class – the dataset presents 
        # only the grape class that is numbered 0, so every line starts with 
        # this "class zero" indicator. The center of the bounding box is the 
        # point (c_x, c_y), represented as float values because this format 
        # normalizes the coordinates by the image dimensions. To get the 
        # absolute position, use (2048 c_x, 1365 c_y). The bounding box 
        # dimensions are given by W and H, also normalized by the image size.
        #
        # Torchvision's Mask R-CNN expects absolute coordinates.
        _, height, width = img.shape

        boxes = []
        for box in wgisd_boxes:
            x1 = box[0] - box[2]/2
            x2 = box[0] + box[2]/2
            y1 = box[1] - box[3]/2
            y2 = box[1] + box[3]/2
            boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
        
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # ou poderíamos usar
        # labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id
        }

        return img, target

    def __len__(self):
        return len(self.imgs)

logging.basicConfig(
    format='%(levelname)s: %(asctime)s %(message)s',
    level = logging.DEBUG)
NUM_CLASSES = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
NUM_EPOCHS = 8
BATCH_SIZE = 1

dataset = WGISDMaskedDataset('./wgisd')
dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True, collate_fn=lambda s: tuple(zip(*s)))
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
#replace head predictor (boxes)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
#replace mask predictor
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(DEVICE)
for param in model.parameters():
    param.requires_grad = True
model.train()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
n_batches = len(dataloader)
for epoch in range(1, NUM_EPOCHS + 1):
    logging.info(f'Current epoch: {epoch} of {NUM_EPOCHS}')
    lossacc= 0.0
    lossmaskacc = 0.0
    for batchid, (images, targets) in enumerate(dataloader, 1):
        images = [image.to(DEVICE) for image in images]
        targets = [ {k : v.to(DEVICE) for k, v in t.items()}for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # #logs
        lossmask=loss_dict['loss_mask'].item()
        lossmaskacc += lossmask
        lossacc += loss.item()
        if (batchid % 25 ==0):
            logging.info(f'lossmask:{lossmask}; loss:{loss.item()}; lossmaskacc:{lossmaskacc}; lossacc: {lossacc};')
    logging.info(f'finalepoch{epoch}: lossmask:{lossmask}; loss:{loss.item()}; lossmaskacc:{lossmaskacc}; lossacc: {lossacc};')
torch.save(model.state_dict(), './models/model0.pt')
logging.info('finished')