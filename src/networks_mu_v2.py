"""Same policy architecture, factorized first MLP affine layer for full mu.

Avoid materializing repeated macro coordinates for every household query. The
sum of macro and micro affine terms equals the original concatenated linear
layer; parameter names/counts are unchanged. Only new v2 runs use this code.
"""
import torch
from networks import PolicyNetwork


class MuPolicyNetwork(PolicyNetwork):
    def forward(self,mu,z,asset,employment):
        if self.kind=='deeponet':return super().forward(mu,z,asset,employment)
        if self.policy_variant!='base':raise ValueError('v2 factorized MLP requires base policy')
        macro=self.macro_inputs(mu,z)
        micro=torch.stack((asset/self.asset_max,employment.to(mu.dtype)),dim=-1)
        first=self.single[0];split=macro.shape[-1]
        macro_term=torch.nn.functional.linear(macro,first.weight[:,:split],first.bias)
        micro_term=torch.nn.functional.linear(micro,first.weight[:,split:],None)
        value=macro_term[:,None,None,:]+micro_term
        for layer in list(self.single)[1:]:value=layer(value)
        return value.squeeze(-1)
