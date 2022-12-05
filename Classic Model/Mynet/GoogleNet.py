#GoogleNet， 并行网络结构。
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self,in_c,out1,out2,out3,out4):
        super(Inception, self).__init__()
        self.path1_part1 = nn.Conv2d(in_c,out1,kernel_size=1)

        self.path2_part1 = nn.Conv2d(in_c,out2[0],kernel_size=1)
        self.path2_part2 = nn.Conv2d(out2[0],out2[1],kernel_size=3, padding=1)

        self.path3_part1 = nn.Conv2d(in_c, out3[0], kernel_size=1)
        self.path3_part2 = nn.Conv2d(out3[0],out3[1], kernel_size=5, padding=2)

        self.path4_part1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.path4_part2 = nn.Conv2d(in_c,out4, kernel_size=1)

    def forward(self, x):
        path1_out = F.relu(self.path1_part1(x))
        path2_out = F.relu(self.path2_part2(F.relu(self.path2_part1(x))))
        path3_out = F.relu(self.path3_part2(F.relu(self.path3_part1(x))))
        path4_out = F.relu(self.path4_part2(self.path4_part1(x)))
        return torch.cat((path1_out,path2_out,path3_out,path4_out), dim=1)

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self,x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   GlobalAvgPool2d())

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.Flatten(), nn.Linear(1024, 10))

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Flatten(), nn.Linear(1024, 10))
X = torch.rand(1, 1, 96, 96)
for blk in net.children():
    X = blk(X)
    print('output shape: ', X.shape)

from mnist_load import load_data_fashion_mnist
batch_size = 32
train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size, resize=96)
lr, num_epochs = 0.001, 3
optimizer = torch.optim.Adam(net.parameters(),lr=lr)

def evaluate_Googlenet(net, test_dataloader, device=None):
    net.eval()
    if device == None and torch.cuda.is_available():
        device = "cuda:0"
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X,y in test_dataloader:
            acc_sum += (net(X.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    net.train()
    return acc_sum/n

def train_Google(net, train_dataloader, test_dataloader, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("trainning on",device)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X,y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
            n+= y.shape[0]
            batch_count += 1
        test_acc = evaluate_Googlenet(net, test_dataloader)
        print("epoch %d, loss %.4f, train acc %.3f, test acc%.3f, time %.1f sec"
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_Google(net, train_dataloader, test_dataloader, batch_size, optimizer, device, 3)



