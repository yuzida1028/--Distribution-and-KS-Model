"""Independent reporting metrics and frozen-grid policy adapter."""
import numpy as np
import torch
from jedc_distribution_v1 import rebin,midpoint_grid


class FrozenGridPolicy(torch.nn.Module):
    """Keep checkpoint input coordinates; rebin each current/continuation mu.

    This is a quadrature/transport sensitivity check, not a finer-input model.
    It refuses to erase positive mass beyond the checkpoint's input domain.
    """
    def __init__(self, network, evaluation_edges, network_edges):
        super().__init__();self.network=network
        self.register_buffer('source_edges',evaluation_edges)
        self.register_buffer('target_edges',network_edges)
        indices=torch.searchsorted(evaluation_edges,network_edges,right=True).clamp(1,len(evaluation_edges)-1)
        self.register_buffer('lo',indices-1);self.register_buffer('hi',indices)
        fraction=(network_edges-evaluation_edges[indices-1])/(evaluation_edges[indices]-evaluation_edges[indices-1])
        self.register_buffer('fraction',fraction.clamp(0,1))

    def forward(self,mu,z,asset,employment):
        if torch.equal(self.source_edges,self.target_edges):return self.network(mu,z,asset,employment)
        if self.source_edges[-1]>self.target_edges[-1]:
            overflow=mu[:,:,1:][:,:,self.source_edges[1:]>self.target_edges[-1]].sum()
            if float(overflow)>1e-12:raise ValueError('checkpoint input adapter would truncate positive mass')
        cdf=torch.cat((torch.zeros_like(mu[:,:,:1]),torch.cumsum(mu[:,:,1:],-1)),-1)
        interpolated=cdf[:,:,self.lo]*(1-self.fraction)+cdf[:,:,self.hi]*self.fraction
        encoded=torch.cat((mu[:,:,:1],torch.diff(interpolated,dim=-1)),dim=-1)
        return self.network(encoded,z,asset,employment)


def quantiles(mu,edges):
    mass=mu.sum(0);cdf=np.cumsum(mass);values=[]
    for q in [.1,.5,.9,.99]:
        i=int(np.searchsorted(cdf,q*mass.sum()).clip(0,len(mass)-1))
        values.append(0. if i==0 else float(edges[i-1]+(q*mass.sum()-cdf[i-1])/max(mass[i],1e-30)*(edges[i]-edges[i-1])))
    return values


def metrics(economy,network,mu,z):
    if len(mu)!=1:raise ValueError('report one common state at a time')
    with torch.no_grad():
        saving,consumption,gap,next_mu=economy.euler_terms(network,mu,z)
        assets,employment=economy.edge_queries(1)
        saving_edges,edge_consumption=economy.policy(network,mu,z,assets,employment)
        K,w,r,tax=economy.prices(mu,z)
    mass=mu.cpu().numpy()[0].astype(float);grid=economy.grid.cpu().numpy().astype(float)
    s=saving.cpu().numpy()[0];c=consumption.cpu().numpy()[0];g=gap.cpu().numpy()[0]
    violation=np.where(s>1e-7,abs(g),np.maximum(-g,0))
    weights=mass.ravel();v=violation.ravel();order=np.argsort(v);cumulative=np.cumsum(weights[order]);total=weights.sum()
    def q(p):return float(v[order[int(np.searchsorted(cumulative,p*total).clip(0,len(order)-1))]])
    e_s=saving_edges.cpu().numpy()[0]
    affine_K=float((mass[:,0]*e_s[:,0]).sum()+(mass[:,1:]*(e_s[:,:-1]+e_s[:,1:])/2).sum())
    kz=int(z[0]);employment_mass=mass.sum(1);labor=employment_mass[1]*economy.hours
    productivity=float(economy.z_values[kz]);Y=productivity*float(K)**economy.alpha*labor**(1-economy.alpha)
    average_c=float((mass*c).sum());midpoint_K=float((mass*s).sum())
    raw_K=float((mass*grid).sum())
    result=dict(K=raw_K,price_K=float(K),price_capital_floor_active=float(raw_K<.2),
        price_labor_floor_active=float(labor<.05),KKT=float(weights@v/total),median=q(.5),P90=q(.9),P99=q(.99),
        max_populated_KKT=float(v[weights>0].max()),min_consumption=float(c.min()),
        min_populated_consumption=float(c[mass>0].min()),min_saving=float(s.min()),
        borrowing_violation_mass=float(mass[s<0].sum()),exact_borrowing_mass=float(mass[s==0].sum()),
        upper_saturation_mass=float(mass[s>=economy.asset_max-1e-6].sum()),
        bound_negative_gap_mass=float(mass[(s<=1e-7)&(g<0)].sum()),
        one_step_midpoint_policy_K=midpoint_K,one_step_affine_policy_K=affine_K,
        capital_policy_quadrature_error=affine_K-midpoint_K,
        next_mass_error=max(abs(float(m.sum())-total) for m in next_mu),
        next_capital_reprojection_error=max(abs(float((m*economy.grid).sum())-affine_K) for m in next_mu),
        next_employment_error=max(float(torch.max(abs(m.sum(2)[0]-m.new_tensor([.1,.9] if nz==0 else [.04,.96])))) for nz,m in enumerate(next_mu)),
        Y=Y,wage=float(w),interest=float(r),tax=float(tax),average_consumption=average_c,
        tax_UI_budget_sanity=float(tax)*float(w)*labor-economy.insurance*float(w)*employment_mass[0],
        resource_budget_midpoint_sanity=Y+(1-economy.delta)*float(K)-average_c-midpoint_K,
        resource_budget_affine_difference=Y+(1-economy.delta)*float(K)-average_c-affine_K,
        edge_min_consumption=float(edge_consumption.min()))
    for e in range(2):
        for label,mask in [('zero',grid==0),('low',(grid>0)&(grid<=5)),('mid',(grid>5)&(grid<=20)),('high',grid>20)]:
            m=mass[e,mask];result['e%d_%s_mass'%(e,label)]=float(m.sum())
            result['e%d_%s_KKT'%(e,label)]=float(m@violation[e,mask]/max(m.sum(),1e-30))
    if not all(np.isfinite(v) for v in result.values()):raise FloatingPointError('nonfinite policy metrics')
    if result['min_consumption']<=0:raise FloatingPointError('nonpositive consumption')
    return result
