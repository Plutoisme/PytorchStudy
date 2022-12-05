import torch
import torch.nn as nn
from mnist_load import load_data_fashion_mnist
import time

def evaluate_accuracy_LeNet(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if torch.cuda.is_available() and device==None:
            device = "cuda:0"
    acc_sum, n =0.0,0
    with torch.no_grad():
        for X,y in data_iter:
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
        n += y.shape[0]
    return acc_sum/n

def train_LeNet(net, train_dataloader, test_dataloader, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on",device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            print(y_hat.shape)
            print(y.shape)
            l = loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
            n+=y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy_LeNet(net, test_dataloader)
        print("epoch %d, loss %.4f, train acc %.3f, test acc%.3f, time %.1f sec"
              %(epoch+1, train_loss_sum/batch_count, train_acc_sum/n, test_acc, time.time()-start))


net = nn.Sequential(
    nn.Conv2d(1,6,kernel_size=5), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6,16, kernel_size=5), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*4*4,120), nn.Sigmoid(),
    nn.Linear(120,84), nn.Sigmoid(),
    nn.Linear(84,10)
)

for layer in net:
    print(layer.__class__.__name__)

batch_size = 128
# 定义dataloader
train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size=batch_size)
lr, num_epochs=0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_LeNet(net, train_dataloader, test_dataloader, batch_size, optimizer, "cuda:0"if torch.cuda.is_available() else "cpu",num_epochs)

