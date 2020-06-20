import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from model import AlexNet, VGG, Inception

batch_size = 16
learning_rate = 0.0001
epochs = 50
num_workers = 1


def loadData():
    train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            ])
    
    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
            ])
    
    train_dir = 'dataset/classify/train'
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_dir = 'dataset/classify/val'
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader, val_dataloader


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for i, data in enumerate(loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
        correct += torch.eq(pred, labels).sum().float().item()
    return correct/total


def train():
    train_dataloader, val_dataloader = loadData()
    pretrained_params = torch.load('VGG_pretrained.pth')
    model = VGG()
    # strict=False 使得预训练模型参数中和新模型对应上的参数会被载入，对应不上或没有的参数被抛弃。
    model.load_state_dict(pretrained_params.state_dict(), strict=False)
    
    if torch.cuda.is_available():
        model.cuda()
        
    # finetune 时冻结XXlayer的参数
#    for p in model.XXlayers.parameters():
#        p.requires_grad = False
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    best_acc = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        steps = 0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]
            steps += 1
        print('epoch:%d loss:%.3f' % (epoch+1, epoch_loss / steps))
        if epoch % 5 == 0:
            val_acc = evaluate(model, val_dataloader)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model, 'best_VGG.pkl')
                torch.save(model.state_dict(), 'best_VGG_params.pkl')
            print('test acc:'.format(val_acc))
                
    print('Finished Training')
    torch.save(model, 'VGG.pkl')
    torch.save(model.state_dict(), 'VGG_params.pkl')
    
    
if __name__ == '__main__':
    train()