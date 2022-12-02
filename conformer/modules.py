import torch
import torch.nn as nn
import torch.nn.init as init

'''
 refer:Understanding the difficulty of training deep feedforward neural networks:
 https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
 https://zhuanlan.zhihu.com/p/490518183
 
 强调一下参数初始化的重要性：比如全零初始化会导致参数的绝对对称问题：每一层神经网络退化成单一神经元。
 
torch.nn.init.xavier_uniform_是一个服从均匀分布的Glorot初始化器，表达式为： U = (-a,a), a = gain*sqrt(6/(fan_in+fan_out))
为什么需要经过Xavier： 通过网络层时，输入和输出的方差尽可能相同。（Internal Covariance Shift，ICS提出在网络前部如果每一层的方差变化很大，后面的
网络很难捕捉到有用特征)

Kaiming初始化：在Xavier的基础上，假设每层网络有一半的神经元被关闭，于是其分布的方差也会变小。
经过验证发现当对初始化值缩小一半时效果最好，故He初始化可以认为是Xavier初始/2的结果。
'''

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True)->None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear(x)
## 方便把转置操作集成到nn.Sequential中
## 在需要导出模型时，将模型里的数据操作使用nn.module代替是个好习惯。
class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return x.transpose(*self.shape)