import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


train_dir = '/data/botml/leaf_dimension_classifier/classed_training_data/'
data_dir = '/data/botml/leaf_dimension_classifier/classifier_training_testdata/'
model_file = '/data/botml/leaf_dimension_classifier/model/classifier/model.pth'
out_dir = '/data/botml/leaf_dimension_classifier/classifier_results_test.csv'

test_transforms = transforms.Compose([transforms.Resize((320,224)),
                                      transforms.ToTensor(),
                                     ])

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


# v2
def get_images():
    #data = datasets.ImageFolder(data_dir, transform=test_transforms)
    fnames = data.imgs
    indices = list(range(len(data)))
    idx = indices
    #from torch.utils.data.sampler import SubsetRandomSampler
    #sampler = SubsetRandomSampler(idx)
    #loader = torch.utils.data.DataLoader(data, 
    #               sampler=sampler, batch_size=num)
    loader = torch.utils.data.DataLoader(data, 
                   batch_size=len(idx))
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
    fn  = fnames[ii][0]
    image = to_pil(images[0][ii])
    pclass = predict_image(image)
    #print(fn, classes[pclass])
    results.loc[len(results)] = [fn, classes[pclass]]
    

    
results.to_csv(out_dir, sep=',')
