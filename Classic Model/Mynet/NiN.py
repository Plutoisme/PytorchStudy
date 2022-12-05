# 1*1的卷积层用来代替全连接层，使得空间信息自然传递到后面的层中去。
import time
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels,out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )
    return blk

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self,x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

ninnet = nn.Sequential(
    nin_block(1,96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96,256, kernel_size=5, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256,384,kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    GlobalAvgPool2d(),
    nn.Flatten())

X = torch.rand(8,1,224,224)
for name, blk in ninnet.named_children():
    X = blk(X)
    print(name, "output shape:", X.shape)

'''
0:(8,1,224,224) -> (8,96,54,54) -> (8,96,54,54) -> (8,96,54,54)
1:(8,96,54,54) -> (8,96,(54-3)//2+1,..)=(8,96,26,26)
2: ..
..
..
..
8: (8,10,5,5) -> (8,10,1,1)
9: (8,10,1,1) -> (8,10)
'''

from mnist_load import load_data_fashion_mnist
batch_size = 32
train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size, resize = 224)

lr, num_epochs = 0.002, 5
optimizer = torch.optim.Adam(ninnet.parameters(), lr=lr)

def evaluate_nin(net, test_dataloader, device = None):
    net.eval()
    if device==None and torch.cuda.is_available():
        device = "cuda:0"
    acc_sum, n =0.0, 0
    with torch.no_grad():
        for X,y in test_dataloader:
            acc_sum +=(net(X.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    net.train()
    return acc_sum/n

def train_nin(net, train_dataloader, test_dataloader, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("trainning on ",device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X,y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            y_hat = ninnet(X)
            l = loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
            n+=y.shape[0]
            batch_count += 1
        test_acc = evaluate_nin(ninnet, test_dataloader)
        print("epoch %d, loss %.4f, train acc %.3f, test acc%.3f, time %.1f sec"
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_nin(ninnet, train_dataloader, test_dataloader, batch_size, optimizer, device, 3)
