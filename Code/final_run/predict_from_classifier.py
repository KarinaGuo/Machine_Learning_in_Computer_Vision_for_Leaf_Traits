import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import csv
from csv import writer
import os
from torchvision import datasets, transforms, models
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("base_dir", help="", type=str,)
parser.add_argument("loop", help="", type=str,)
args = parser.parse_args()

if args.loop == "main":
	train_dir = os.path.join(args.base_dir, "classifier_training_data/")
	data_dir = os.path.join(args.base_dir, "temp_pred_leaf/") 
    
if args.loop == "model":
	train_dir = os.path.join(args.base_dir, "input_classifier_data/")
	data_dir = os.path.join(args.base_dir, "pred_leaf/")
	
model_file = os.path.join(args.base_dir, "model/classifier/model.pth")
out_dir = os.path.join(args.base_dir, "classifier_results_test.csv")
  
test_transforms = transforms.Compose([transforms.Resize((500,500)), transforms.ToTensor()])
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	
model=torch.load(model_file)
model.eval()
  
data_train = datasets.ImageFolder(train_dir, transform=test_transforms)
classes = data_train.classes
  
def predict_image(image):
	image_tensor = test_transforms(image).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input = image_tensor
	input = input.to(device)
	output = model(input)
	index = output.data.cpu().numpy().argmax()
	return index
  
if args.loop == "main":
	def get_images():
		fnames = data.imgs
		indices = list(range(len(data)))
		idx = indices
		loader = torch.utils.data.DataLoader(data, batch_size=len(idx))
		dataiter = iter(loader)
		images = dataiter.next()
		return images, fnames    
	data = datasets.ImageFolder(data_dir, transform=test_transforms)
	to_pil = transforms.ToPILImage()
	images, fnames = get_images()
	colnames = ['filename','pr_class']
	results = pd.DataFrame(columns=colnames)
	indices = list(range(len(data)))
	for ii in range(len(indices)):
		fn  = os.path.basename(fnames[ii][0])
		image = to_pil(images[0][ii])
		pclass = predict_image(image)
		results.loc[len(results)] = [fn, classes[pclass]]
		results.to_csv(out_dir, index=False, mode='wb')

if args.loop == "model":
	def get_images():
		classes = data.classes
		fnames = data.imgs
		indices = list(range(len(data)))
		idx = indices
		loader = torch.utils.data.DataLoader(data, batch_size=len(idx))
		dataiter = iter(loader)
		images, labels = dataiter.next()
		return images, labels, fnames
	data = datasets.ImageFolder(data_dir, transform=test_transforms)
	to_pil = transforms.ToPILImage()
	images, labels, fnames = get_images()
	colnames = ['file_name', 'ind_pr', 'gt_class', 'pr_class']
	results = pd.DataFrame(columns=colnames)
	indices = list(range(len(data)))
	for ii in range(len(indices)):
		temp_fn  = os.path.basename(fnames[ii][0])
		new_str = temp_fn.split("_")
		fn = f"{new_str[0]}.jpg"
		ind_pr = new_str[1]
		image = to_pil(images[ii])
		pclass = predict_image(image)
		lclass = labels[ii].item()
		results.loc[len(results)] = [fn, ind_pr, classes[lclass], classes[pclass]]
		print(fn, ind_pr, classes[lclass], classes[pclass])
		results.to_csv(out_dir, index=False)
