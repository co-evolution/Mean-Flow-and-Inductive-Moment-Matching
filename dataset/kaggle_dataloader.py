import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

IMAGENET_PATH = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/"
TRAIN_DIR = os.path.join(IMAGENET_PATH, "train")
VAL_DIR = os.path.join(IMAGENET_PATH, "val")   
TEST_DIR = os.path.join(IMAGENET_PATH, "test")   


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(), 
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])



def get_train_loader(batch_size,num_workers):
    train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    return train_loader

def get_test_loader(batch_size,num_workers):
    test_dataset= ImageFolder(TEST_DIR, transform=transform)
    train_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    return train_loader