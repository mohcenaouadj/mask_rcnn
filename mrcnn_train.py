import random
import numpy as np
import os
import cv2 as cv
import torch.utils.data
import torchvision.models.segmentation
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



# Configs
batch_size = 2
imageSize = [600,600]

device = torch.device('cpu')

trainDir = '/home/mohcenaouadj/Téléchargements/LabPicsMedical/Train'
imgs = []
for pth in os.listdir(trainDir):
	imgs.append(trainDir+"/"+pth+"/")

print('Number of directories : ',len(imgs))


def loadData():
	batch_Imgs = []
	batch_Data = []

	for i in range(batch_size):
		idx=random.randint(0,len(imgs)-1)
		img = cv.imread(os.path.join(imgs[idx], "Image.jpg"))
		img = cv.resize(img, imageSize, cv.INTER_LINEAR)


		maskDir = os.path.join(imgs[idx], "Vessels")
		masks = []
		for mskName in os.listdir(maskDir):
			vesMask = cv.imread(maskDir+'/'+mskName, 0)
			vesMask = (vesMask > 0).astype(np.uint8)
			vesMask = cv.resize(vesMask, imageSize, cv.INTER_LINEAR)
			masks.append(vesMask)

		num_objs = len(masks)
		if num_objs == 0 : 
			return loadData()

		boxes = torch.zeros([num_objs, 4], dtype = torch.float32)

		for i in range(num_objs):
			x,y,w,h = cv.boundingRect(masks[i])
			boxes[i] = torch.tensor([x, y, x+w, y+h])

		masks = np.array(masks)
		masks = torch.as_tensor(masks, dtype = torch.uint8)
		img = torch.as_tensor(img, dtype = torch.float32)

		data = {}
		data["boxes"] = boxes
		data["labels"] = torch.ones((num_objs,), dtype = torch.int64)
		data["masks"] = masks

		batch_Imgs.append(img)
		batch_Data.append(data)


	batch_Imgs=torch.stack([torch.as_tensor(d) for d in batch_Imgs],0)
	batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
  	

	return batch_Imgs, batch_Data


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)

in_features = model.roi_heads.box_predictor.cls_score.in_features 
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=2)

model.to(device)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
model.train()

for i in range(200):
	print('i in loop equals : ',i)
	images, targets = loadData()

	images = list(image.to(device) for image in images)
	targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

	optimizer.zero_grad()
	loss_dict = model(images, targets)

	losses = sum(loss for loss in loss_dict.values())
	losses.backward()
	optimizer.step()
	print(i,'loss:', losses.item())
	if i%150==0:
		torch.save(model.state_dict(), str(i)+".torch")