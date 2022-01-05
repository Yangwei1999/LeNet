import torch

from net import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    my_model = My_LeNet()
    a = torch.zeros((10,1,28,28))
    print(a.shape)
    out = my_model(a)
    print(out.shape)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
