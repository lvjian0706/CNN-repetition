import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from model import AlexNet, VGG, Inception


def loadData():
    test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
            ])
    
    test_dir = 'dataset/classify/test'
    test_dataset = datasets.ImageFolder(test_dir, transforms=test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
    return test_dataloader

def loadNet():
    model = torch.load('best_VGG.pkl')
    return model


def test():
    test_dataloader = loadData()
    model = loadNet()
    model.eval()
    correct = 0
    total = len(test_dataloader.dataset)
    for i, data in enumerate(test_dataloader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
        correct += torch.eq(pred, labels).sum().float().item()
    return correct/total
    

if __name__ == '__main__':
    test()