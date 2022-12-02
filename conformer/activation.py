import torch
import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * inputs.sigmoid()

#GLU：https://arxiv.org/abs/1612.08083
class GLU(nn.Module):
    def __init__(self, dim:int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs:torch.Tensor)->torch.Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

'''
门控线性单元GLU应当与循环神经网络中的GRU区分开。
GLU更容易进行梯度的传播，不容易造成梯度消失或者爆炸，计算资源也需求较少。
在这里没有进行传统GLU卷积层的实现。

'''
if __name__ == "__main__":
    print("unit activation test:")
    input1 = torch.randn(10,10,80)
    input2 = torch.randn(10,6,10)
    swish = Swish()
    glu = GLU(1)
    output1 = swish(input1)
    output2 = glu(input2)
    print(output1.shape)
    print(output2.shape)