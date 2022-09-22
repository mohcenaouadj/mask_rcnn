import random
import numpy as np
import os
import cv2 as cv
import torch.utils.data
import torchvision.models.segmentation
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


imgPath = 'LabPicsMedical/Test/8Eval_IVbags/Image.jpg'
imageSize = [600,600]
device = torch.device('cpu')

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained= True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes= 2)

model.load_state_dict(torch.load("150.torch"))
model.to(device)# move model to the right devic
model.eval()


images = cv.imread(imgPath)
images = cv.resize(images, imageSize, cv.INTER_LINEAR)
images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
images=images.swapaxes(1, 3).swapaxes(2, 3)
images = list(image.to(device) for image in images)

with torch.no_grad():
    pred = model(images)


im= images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)
im2 = im.copy()
msk = 0
showed = 0
for i in range(len(pred[0]['masks'])):
	msk = pred[0]['masks'][i,0].detach().cpu().numpy()
	showed = pred[0]['scores'][i].detach().cpu().numpy()
	if showed>0.8 :
		im2[:,:,0][msk>0.5] = random.randint(0,255)
		im2[:, :, 1][msk > 0.5] = random.randint(0,255)
		im2[:, :, 2][msk > 0.5] = random.randint(0, 255)

cv.imshow(str(showed), np.hstack([im,im2]))
cv.waitKey()