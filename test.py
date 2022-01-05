import torch
from torch.utils.data import DataLoader

from net import My_LeNet
from torch.autograd import Variable
from torchvision import datasets,transforms
from torchvision.transforms import ToPILImage
# 用真实图片看可视化结果

data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data',train=True,transform=data_transform,download=True)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)

test_dataset = datasets.MNIST(root='./data',train=False,transform=data_transform,download=True)
test_dataloader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)

# 如果FPU
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
# 调用网络模型

model = My_LeNet().to(device)

model.load_state_dict(torch.load('save_model/best.pth'))

# 获取结果
classes = [str(i) for i in range(10)]

# 把tensor转换成图片  方便可视化

show = ToPILImage()

# 进入验证

for i in range(10):
    X,y = test_dataset[i]
    show(X).show()
    X = Variable(torch.unsqueeze(X,dim=0),requires_grad = False).to(device)
    with torch.no_grad():
        pred = model(X)

        predicted,actual = classes[torch.argmax(pred[0])],classes[y]

        print('predicted:{}.actual:{}'.format(predicted,actual))




