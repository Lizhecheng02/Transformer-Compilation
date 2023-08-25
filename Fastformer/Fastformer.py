import einops
from einops import rearrange
import torch
import torch.nn as nn


class Fastformer(nn.Module):
    def __init__(self, dim=3, decode_dim=16):
        super(Fastformer, self).__init__()
        self.to_qkv = nn.Linear(dim, decode_dim * 3, bias=False)
        self.weight_q = nn.Linear(dim, decode_dim, bias=False)
        self.weight_k = nn.Linear(dim, decode_dim, bias=False)
        self.weight_v = nn.Linear(dim, decode_dim, bias=False)

        self.weight_gamma = nn.Linear(decode_dim, decode_dim, bias=False)
        self.weight_alpha = nn.Parameter(torch.randn(decode_dim))
        self.weight_beta = nn.Parameter(torch.randn(decode_dim))
        self.scale_factor = decode_dim ** -0.5

    def forward(self, x, mask=None):
        query = self.weight_q(x)
        key = self.weight_k(x)
        value = self.weight_v(x)
        b, n, d = query.shape

        mask_value = torch.finfo(x.dtype).max
        mask = rearrange(mask, 'b n -> b () n')

        alpha_weight = (torch.mul(query, self.weight_alpha) *
                        self.scale_factor).masked_fill(~mask, mask_value)
        alpha_weight = torch.softmax(alpha_weight, dim=-1)
        """
        根据给定的代码torch.mul(query, self.weight_alpha)，它执行了张量query与张量self.weight_alpha的逐元素相乘操作。根据广播规则，当两个张量的形状不完全匹配时，PyTorch会自动进行广播。

        在这种情况下，self.weight_alpha的形状是（64，），它将被广播为与query的形状（2，10，64）相匹配。广播操作会将self.weight_alpha的维度扩展为（1，1，64），然后将其复制为与query相同的形状（2，10，64）。

        因此，输出的张量形状将与query的形状相同，即（2，10，64）。
        """
        print("alpha_weight shape:", alpha_weight.shape)
        global_query = query * alpha_weight
        global_query = torch.einsum('b n d -> b d', global_query)
        print("global_query shape:", global_query.shape)

        repeat_global_query = einops.repeat(
            global_query, 'b d -> b copy d', copy=n)
        print("repeat_blobal shape:", repeat_global_query.shape)
        p = repeat_global_query * key
        print("p shape:", p.shape)
        beta_weight = (torch.mul(p, self.weight_beta) *
                       self.scale_factor).masked_fill(~mask, mask_value)
        beta_weight = torch.softmax(beta_weight, dim=-1)
        print("beta_weight shape:", beta_weight.shape)
        global_key = p * beta_weight
        global_key = torch.einsum('b n d -> b d', global_key)
        print("global_key shape:", global_key.shape)

        key_value_interaction = torch.einsum(
            'b j, b n j -> b n j', global_key, value)
        key_value_interaction = self.weight_gamma(key_value_interaction)
        result = key_value_interaction + query

        return result


if __name__ == "__main__":
    model = Fastformer(dim=3, decode_dim=64)
    x = torch.randn(2, 10, 3)
    mask = torch.ones(1, 64).bool()
    result = model(x, mask)
    print("result shape:", result.shape)
