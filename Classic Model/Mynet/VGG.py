'''
VGG是第一个使用重复元素的网络， 其提出了可以通过重复使用简单的基础块来构造深度模型的思路。
'''
import time
import torch
import torch.nn as nn
import torch.optim as optim



def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*blk)

conv_arch=((1,1,64),(1,64,128),(2,128,256),(2,256,512),(2,512,512)) # 5个vgg block
fc_features = 512*7*7 # 经过5个vgg_block， 宽高会减半五次： 224/2^5 = 7
fc_hidden_units = 4096 # 定义隐藏层数

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_"+str(i+1), vgg_block(num_convs, in_channels, out_channels))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(fc_features, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units,10)
                                       ))
    return net

net = vgg(conv_arch, fc_features, fc_hidden_units)
X = torch.rand(1,1,224,224)

for name, blk in net.named_children():
    X = blk(X)
    print(name, "output shape:", X.shape)

# 减少网络中卷积核的out_channels, 来减少参数实现demo：
ratio = 4
small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio),
                   (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
net = vgg(small_conv_arch, fc_features//ratio, fc_hidden_units//ratio)

from mnist_load import load_data_fashion_mnist

batch_size = 16
train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size,resize=224)

lr, num_epochs = 0.001, 3
optimizer = torch.optim.Adam(net.parameters(),lr=lr)

# 编写测试函数，在训练主函数中调用：
def evaluate_accuracy_vgg(net, test_dataloader, device =None):
    net.eval()
    if device==None and torch.cuda.is_available():
        device = "cuda:0"
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X,y in test_dataloader:
            acc_sum += (net(X.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    net.train()
    return acc_sum/n

# 编写训练主函数：

def train_vgg(net, train_dataloader, test_dataloader, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("trainning on ",device)
    loss = torch.nn.CrossEntropyLoss()
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
            n+=y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy_vgg(net, test_dataloader)
        print("epoch %d, loss %.4f, train acc %.3f, test acc%.3f, time %.1f sec"
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_vgg(net, train_dataloader, test_dataloader, batch_size, optimizer, device, 3)