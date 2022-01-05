import torch
from torch import nn
from torch.utils.data import DataLoader

from net import My_LeNet

from torch.optim import lr_scheduler
from torchvision import datasets,transforms

import os

# 数据转换为tensor格式

data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data',train=True,transform=data_transform,download=True)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)

test_dataset = datasets.MNIST(root='./data',train=False,transform=data_transform,download=True)
test_dataloader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)


# 如果FPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 调用网络模型

model = My_LeNet().to(device)

# 定义损失函数

loss_fn = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)

# 学习率 每隔10轮 变为原来的0.1

lr_scheduler1 = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)


# 定义训练函数

def train(model,train_dataloader,optimizer,loss_fn):
    loss = 0
    accuracy  = 0
    n = 0
    for batch,(X,y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        X,y = X.to(device),y.to(device)
        output = model(X)
        cur_loss = loss_fn(output,y)

        cur_loss.backward()
        optimizer.step()

        _,pred = torch.max(output,axis=1)

        cur_acc = torch.sum(y==pred)/output.shape[0]
        loss += cur_loss.item()
        accuracy+=cur_acc.item()
        n = n + 1

    print('train_loss:',loss/n)
    print('train_acc:',accuracy/n)



def val(model,test_dataloader,loss_fn):
    model.eval()

    loss = 0
    accuracy = 0
    n = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            accuracy += cur_acc.item()
            n = n + 1
        print('val_loss:', loss / n)
        print('val_acc:', accuracy / n)

        return accuracy/n





epoch = 50
min_acc = 0
for i in range(epoch):
    print('epoch : {}----------------------'.format(i))
    train(model,train_dataloader,optimizer,loss_fn)
    a = val(model,test_dataloader,loss_fn)
    # serach the best weights
    if a>min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.makedirs(folder)
        min_acc = a

        print('save best model')
        torch.save(model.state_dict(),'save_model/best.pth')

    if i == epoch - 1:
        print('save last model')
        torch.save(model.state_dict(),'save_model/last.pth')

print('done')



