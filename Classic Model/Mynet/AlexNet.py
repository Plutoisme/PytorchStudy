import time
import torch
from torch import nn, optim
import torchvision
from mnist_load import load_data_fashion_mnist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.convpart = nn.Sequential(
            nn.Conv2d(1,128,11,4),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(128,256,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Flatten()
        )
        self.fcpart = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )

    def forward(self,img):
        feature = self.convpart(img)
        output = self.fcpart(feature)
        return output

alexnet = AlexNet()
print(alexnet)
'''
with torch.no_grad():
    input = torch.randn(32,1,224,224)
    output = alexnet(input)
    print(output.shape)
'''

lr, num_epochs = 0.001, 3
optimizer = torch.optim.Adam(alexnet.parameters(),lr=lr)

def evaluate_accuracy_AlexNet(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if torch.cuda.is_available() and device==None:
            device = "cuda:0"
    acc_sum, n = 0.0,0
    with torch.no_grad():
        for X,y in data_iter:
            acc_sum += (net(X.to(device)).argmax(dim=1)== y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    net.train()
    return acc_sum/n


def train_AlexNet(net, train_dataloader, test_dataloader, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ",device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X,y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            # y.shape = (32,10)
            y_hat = net(X)
            l = loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
            n+=y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy_AlexNet(net, test_dataloader)
        print("epoch %d, loss %.4f, train acc %.3f, test acc%.3f, time %.1f sec"
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

batch_size = 64
train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size, resize=224)
train_AlexNet(alexnet, train_dataloader, test_dataloader, batch_size, optimizer, device, 3)