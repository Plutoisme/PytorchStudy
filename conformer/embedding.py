import math
import torch
import torch.nn as nn

#PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
#PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))

# Relative Positional Encoding.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 10000) -> None:
        super(PositionalEncoding,self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False) # size:(10000,512)
        position = torch.arange(0,max_len, dtype=torch.float).unsqueeze(1) # size:(10000) -> size:(10000,1)
        #div_term = exp[2j*(-ln10000/d)]=(exp[-ln(10000)])^(2j/d)
        div_term = torch.exp(torch.arange(0,d_model, 2).float() * -(math.log(10000.0)/d_model)) # size:(256)
        # broadcast:
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #size:(10000,512)->size:(1,10000,512)
        # register_buffer 定义的一组参数，在训练中不能被更新。
        self.register_buffer("pe", pe)

    def forward(self, length:int) -> torch.Tensor:
        return self.pe[:,:length]

if __name__ == "__main__":
    print("unit code embedding test:")
    positionalencoding = PositionalEncoding()
    pe = positionalencoding(20)
    print(pe.shape)



