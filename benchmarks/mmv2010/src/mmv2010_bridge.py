"""MMV table bridge into the unchanged common interval Euler/KKT evaluator."""
import numpy as np
import torch
from mmv2010_jedc import PolicyTable
from jedc_distribution_v1 import IntervalEconomy


class TorchMMVPolicy(torch.nn.Module):
    def __init__(self,path):
        super().__init__();self.policy_table=PolicyTable.load(path)

    def choices(self,economy,mu,z,assets,employment):
        K=(mu*economy.grid).sum((1,2)).detach().cpu().numpy()[:,None,None]
        choices=self.policy_table.query(K,z.detach().cpu().numpy()[:,None,None],
            assets.detach().cpu().numpy().astype(float),employment.detach().cpu().numpy())
        cash=economy.cash(mu,z,assets,employment)
        upper=torch.minimum((cash-1e-4).clamp_min(0),cash.new_full(cash.shape,economy.asset_max))
        raw=torch.tensor(choices,dtype=mu.dtype,device=mu.device)
        saving=torch.minimum(raw,upper)
        return saving,cash-saving


class MMVIntervalEconomy(IntervalEconomy):
    def policy(self,network,mu,z,assets,employment):
        if isinstance(network,TorchMMVPolicy):return network.choices(self,mu,z,assets,employment)
        return super().policy(network,mu,z,assets,employment)


def perceived_kkt(economy,network,mu,z):
    policy=network.policy_table;zz=int(z[0]);K=float((mu*economy.grid).sum())
    a,e=economy.queries(1)
    with torch.no_grad():saving,c=economy.policy(network,mu,z,a,e)
    s=saving.cpu().numpy()[0];cons=c.cpu().numpy()[0];expected=np.zeros_like(s)
    forecast=np.clip(policy.predict_K(K,zz),policy.capitals[0],policy.capitals[-1])
    from mmv2010_jedc import cash,prices
    for nz in (0,1):
        _,r,_=prices(policy.cal,forecast,nz)
        for ne in (0,1):
            later=policy.query(forecast,nz,s,ne)
            cp=cash(policy.cal,forecast,nz,s,ne)-later
            expected+=np.asarray(policy.cal['joint_transition'])[2*zz:2*zz+2,2*nz+ne][:,None]*(1+r)/cp
    gap=1-economy.beta*expected*cons
    violation=np.where(s>1e-7,abs(gap),np.maximum(-gap,0))
    return float((mu.cpu().numpy()[0]*violation).sum())
