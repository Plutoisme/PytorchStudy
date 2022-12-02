import torch




def _relative_shift(pos_score: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
    zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
    padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

    padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
    pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

    return pos_score

a = torch.randn(10,16,20,20)
b = _relative_shift(a)
print(a.shape)
print(b.shape)
print(a==b.transpose(2,3))