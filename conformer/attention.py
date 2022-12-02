import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .embedding import PositionalEncoding
from .modules import Linear

'''
d_model(int): 词向量的维度
num_heads(int): 多头注意力有多少个头
dropout_p(float): ..

query(batch,time,dim): 
query(batch, time, dim): Tensor containing query vector
key(batch, time, dim): Tensor containing key vector
value(batch, time, dim): Tensor containing value vector
pos_embedding(batch, time, dim): Positional embedding tensor
mask(batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

outputs: Tensor produces by relative multi head attention module.



'''

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self,d_model:int = 512, num_heads:int = 16, dropout_p:float = 0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model does not match num_heads!"
        self.d_head = int(d_model/num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias = False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(self,query: torch.Tensor, key: torch.Tensor, value:torch.Tensor, pos_embedding: torch.Tensor,
                mask: Optional[torch.Tensor] = None)->torch.Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size,-1,self.num_heads,self.d_head)   # size(batch_size, seq_length, num_heads, d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0,2,1,3)
        # for Matrix Multiply, size(batch_size, num_heads, seq_length,d_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0,2,1,3)
        # Assume:batch_size = 10, seq_length=20, d_model=512; pos_embedding.size: (10,20,512) -> (10,20,16,32)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1,2), key.transpose(2,3))  # content_score.size: (10,16,20,20)
        # 为什么这里还需要位置得分呢，位置得分基于位置编码， 我们的注意力不应仅限于内容，否则位置信息就被忽略掉了。
        pos_score = torch.matmul((query + self.v_bias).transpose(1,2), pos_embedding.permute(0,2,3,1)) # pos_score.size: (10,16,20,20)
        # 用到了Transformer-XL提出的Relative Positional Embedding方法？：
        #https: // zhuanlan.zhihu.com / p / 344604604
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.maskerd_fill(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score:torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score],dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:,:,1:].view_as(pos_score)

        return pos_score

class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads:int, dropout_p:float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        '''
        BatchNorm把一个batch内的所有样本同一 通道 的 所有特征 视为一个分布，并进行标准化。
        LayerNorm把一个样本的 所有向量 视作同一个分布， 并进行标准化。
        BatchNorm因此更适用于CV， LayerNorm因此更适合NLP。
        '''
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
    # mask: 让padding(不够长补0)的部分不参与attention操作
    # mask: 生成当前词语的概率分布时，让程序不会注意到这个词背后的部分
    def forward(self, inputs:torch.Tensor, mask:Optional[torch.Tensor]=None):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        # Assume:batch_size = 10, seq_length=20, d_model=512; pos_embedding.size: (1,20,512) -> (10,20,512)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1) # 沿着batch中每个样本复制一次位置编码
        # 前面定义LayerNorm是在词这个维度上进行的，使得所有词向量的分布标准化。
        inputs = self.layer_norm(inputs)
        # 输入到RelativeMultiSelfAttentionModule的query， key， value都是input， 初始化工作由里面的W矩阵完成。
        outputs = self.attention(inputs, inputs, inputs, pos_embedding = pos_embedding, mask=mask)

        return self.dropout(outputs)





