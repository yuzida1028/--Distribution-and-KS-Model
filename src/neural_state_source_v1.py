"""Frozen neural-policy state source. No traditional policy or policy labels.

This supplies endogenous states from an existing approximate neural policy,
not an equilibrium certificate. Only-K and full-mu archived weights are allowed.
For legacy full-mu weights only, midpoint mass is converted to their old input
grid explicitly; the new state evolution still uses independent atom/intervals.
"""
import json
from pathlib import Path
import numpy as np
import torch
from networks import PolicyNetwork
from jedc_distribution_v1 import IntervalEconomy, midpoint_grid, sha

class NeuralStateSource:
    def __init__(self, config_path, checkpoint_path, kind, calibration_path, edges, device='cpu'):
        self.config=json.loads(Path(config_path).read_text(encoding='utf-8'))
        self.device=torch.device(device)
        dtype=torch.float32
        old_grid=torch.linspace(0,self.config['asset_max'],self.config['asset_points'],dtype=dtype,device=self.device)
        h=self.config['hidden'] if kind=='deeponet' else self.config.get('mlp_hidden',self.config['hidden'])
        self.network=PolicyNetwork(kind,len(old_grid),self.config['asset_max'],h,self.config['basis'],
            macro_features=self.config.get('macro_features','none'),grid=old_grid).to(self.device)
        raw=torch.load(checkpoint_path,map_location=self.device)
        self.network.load_state_dict(raw.get('network',raw));self.network.eval()
        self.economy=IntervalEconomy(calibration_path,torch.tensor(edges,dtype=dtype,device=self.device),self.device,
            policy_mode=self.config.get('policy_mode','consumption_softplus'))
        self.old_grid=old_grid;self.edges=np.asarray(edges);self.mu=None
        self.capitals=np.array([0.,float(edges[-1])]);self.assets=self.capitals.copy()
        project=Path(__file__).resolve().parents[1]
        self.provenance=dict(status='frozen_approximate_neural_policy_state_source_not_equilibrium',
            provider='neural',model_kind=kind,config=str(Path(config_path).relative_to(project)).replace('\\','/'),config_sha256=sha(config_path),
            checkpoint=str(Path(checkpoint_path).relative_to(project)).replace('\\','/'),checkpoint_sha256=sha(checkpoint_path),
            macro_features=self.config.get('macro_features','none'),old_encoder_grid_points=len(old_grid),
            interpretation='No policy labels or Young dependency. New models share these states, not the old weights. State-source bias reported separately.')

    def forward(self,mu,z,assets,employment,economy):
        # Network input grid must match legacy weights; K comes from true new mu.
        # For capital_only, any exact-mean two-point mass supplies exactly K to
        # the legacy macro feature function. Residual prices always use real mu.
        capital=(mu*economy.grid).sum((1,2))
        if self.config.get('macro_features')=='capital_only':
            encoded=torch.zeros((len(mu),2,len(self.old_grid)),device=self.device,dtype=mu.dtype)
            group=mu.sum(2)
            fraction=capital/self.config['asset_max']
            encoded[:,:,0]=group*(1-fraction[:,None]);encoded[:,:,-1]=group*fraction[:,None]
        else:
            encoded=torch.zeros((len(mu),2,len(self.old_grid)),device=self.device,dtype=mu.dtype)
            x=economy.grid
            hi=torch.searchsorted(self.old_grid,x).clamp(1,len(self.old_grid)-1);lo=hi-1
            w=((x-self.old_grid[lo])/(self.old_grid[hi]-self.old_grid[lo])).clamp(0,1)
            encoded.scatter_add_(2,lo[None,None,:].expand_as(mu),mu*(1-w))
            encoded.scatter_add_(2,hi[None,None,:].expand_as(mu),mu*w)
        raw=self.network(encoded,z,assets,employment)
        cash=economy.cash(mu,z,assets,employment)
        upper=torch.minimum((cash-1e-4).clamp_min(0),torch.full_like(cash,economy.asset_max))
        if self.config.get('policy_mode')=='consumption_softplus':s=torch.minimum((cash-torch.nn.functional.softplus(raw)-1e-4).clamp_min(0),upper)
        else:
            share=(.5+raw/4).clamp(0,1) if self.config.get('policy_mode')=='hard_sigmoid' else torch.sigmoid(raw)
            s=share*upper
        return s,cash-s

    def choices(self,assets,K,z):
        if self.mu is None:raise ValueError('current mu must be supplied')
        with torch.no_grad():
            mu=torch.as_tensor(self.mu,dtype=torch.float32,device=self.device)[None]
            x=torch.as_tensor(assets,dtype=torch.float32,device=self.device)[None,None,:].expand(1,2,-1)
            e=torch.arange(2,device=self.device)[None,:,None].expand_as(x)
            s,_=self.forward(mu,torch.tensor([z],device=self.device),x,e,self.economy)
            return s[0].cpu().numpy().astype(float)

    def adapter(self,economy):
        return lambda net,mu,z,a,e:self.forward(mu,z,a,e,economy)
