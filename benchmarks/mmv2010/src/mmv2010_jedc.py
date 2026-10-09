"""MMV2010 grid Euler + log-linear KS ALM, independently ported 2026-09-29.

Source: Maliar/Maliar/Valli JEDC34(2010)42-49, Algorithm1 and public
MAIN.M/INDIVIDUAL.m/AGGREGATE_ST.m. New Python/SciPy implementation,
not bitwise MATLAB replication. See docs/MMV2010_JEDC_README.md.
"""
import json
import numpy as np
from scipy.interpolate import RectBivariateSpline


def prices(cal, K, z):
    K=np.asarray(K,dtype=float)
    if np.any(K<=0) or not np.isfinite(K).all():raise ValueError('invalid aggregate capital')
    u=np.asarray(cal['unemployment_rates'])[np.asarray(z,dtype=int)]
    labor=(1-u)*cal['hours_if_employed'];prod=np.asarray(cal['z_values'])[np.asarray(z,dtype=int)]
    wage=(1-cal['alpha'])*prod*(K/labor)**cal['alpha']
    net=cal['alpha']*prod*(K/labor)**(cal['alpha']-1)-cal['delta']
    tax=cal['unemployment_insurance_rate']*u/labor
    return wage,net,tax


def cash(cal,K,z,asset,employment):
    w,r,tax=prices(cal,K,z)
    income=np.where(np.asarray(employment)==1,(1-tax)*cal['hours_if_employed']*w,
                    cal['unemployment_insurance_rate']*w)
    return (1+r)*asset+income


class PolicyTable:
    def __init__(self,cal,assets,capitals,table,coefficients,floor=1e-4):
        self.cal=cal;self.assets=np.asarray(assets);self.capitals=np.asarray(capitals)
        self.table=np.asarray(table);self.coefficients=np.asarray(coefficients);self.floor=floor
        if self.table.shape!=(2,2,len(capitals),len(assets)):raise ValueError('policy dimensions')
        if not np.isfinite(self.table).all():raise ValueError('nonfinite policy')
        self.splines=[[RectBivariateSpline(capitals,assets,self.table[z,e],kx=3,ky=3,s=0)
                       for e in (0,1)] for z in (0,1)]
        self.audit=dict(queries=0,negative_interpolants=0,above_cap_interpolants=0,
                        consumption_floor_projections=0,capital_out_of_domain=0)

    def predict_K(self,K,z):
        value=np.exp(self.coefficients[np.asarray(z),0]+self.coefficients[np.asarray(z),1]*np.log(K))
        return value

    def query(self,K,z,asset,employment):
        K,z,asset,employment=np.broadcast_arrays(K,z,asset,employment)
        if np.any((K<self.capitals[0]-1e-9)|(K>self.capitals[-1]+1e-9)):
            self.audit['capital_out_of_domain']+=int(np.sum((K<self.capitals[0])|(K>self.capitals[-1])))
            raise ValueError(f'K outside frozen [{self.capitals[0]},{self.capitals[-1]}]; no silent clipping')
        if np.any((asset<0)|(asset>self.assets[-1]+1e-8)):raise ValueError('household query outside policy asset grid')
        if np.any(~np.isin(z,[0,1])) or np.any(~np.isin(employment,[0,1])):raise ValueError("invalid employment/productivity")
        out=np.empty(K.shape)
        for zz in (0,1):
            for ee in (0,1):
                mask=(z==zz)&(employment==ee)
                if mask.any():out[mask]=self.splines[zz][ee].ev(K[mask],asset[mask])
        if not np.isfinite(out).all():raise FloatingPointError('nonfinite cubic interpolant')
        resources=cash(self.cal,K,z,asset,employment)
        upper=np.minimum(self.assets[-1],np.maximum(resources-self.floor,0))
        self.audit['queries']+=out.size
        self.audit['negative_interpolants']+=int(np.sum(out<0))
        self.audit['above_cap_interpolants']+=int(np.sum(out>self.assets[-1]))
        self.audit['consumption_floor_projections']+=int(np.sum(out>upper))
        return np.minimum(np.maximum(out,0),upper)

    def save(self,path):
        np.savez_compressed(path,assets=self.assets,capitals=self.capitals,table=self.table,
            coefficients=self.coefficients,calibration_json=json.dumps(self.cal),consumption_floor=self.floor)

    @classmethod
    def load(cls,path):
        with np.load(path) as d:
            return cls(json.loads(str(d['calibration_json'])),d['assets'].copy(),d['capitals'].copy(),
                       d['table'].copy(),d['coefficients'].copy(),float(d['consumption_floor']))


def solve_individual(cal,cfg,coefficients,table=None,progress=None):
    assets=(np.linspace(0,1,cfg['asset_points'])**cfg['asset_grid_power'])*cfg['asset_max']
    capitals=np.linspace(cfg['capital_min'],cfg['capital_max'],cfg['capital_points'])
    A=assets[None,:];K=capitals[:,None]
    resources=np.stack([np.stack([cash(cal,K,z,A,e)+np.zeros((len(capitals),len(assets)))
                                 for e in (0,1)]) for z in (0,1)])
    upper=np.minimum(cfg['asset_max'],np.maximum(resources-cfg['consumption_floor'],0))
    if table is None:table=np.minimum(np.broadcast_to(.9*A,resources.shape),upper).copy()
    transition=np.asarray(cal['joint_transition']);last_error=None;counts={}
    for iteration in range(1,cfg['individual_max_iterations']+1):
        policy=PolicyTable(cal,assets,capitals,table,coefficients,cfg['consumption_floor'])
        new=np.empty_like(table)
        for z in (0,1):
            predicted=policy.predict_K(K,z)
            counts['perceived_forecast_clips']=counts.get('perceived_forecast_clips',0)+int(np.sum((predicted<capitals[0])|(predicted>capitals[-1])))
            predicted=np.clip(predicted,capitals[0],capitals[-1])
            for e in (0,1):
                saving=table[z,e];expected=np.zeros_like(saving)
                for nz in (0,1):
                    for ne in (0,1):
                        later=policy.query(predicted,nz,saving,ne)
                        consumption=cash(cal,predicted,nz,saving,ne)-later
                        _,r,_=prices(cal,predicted,nz)
                        expected+=transition[z*2+e,nz*2+ne]*(1+r)/consumption
                desired=resources[z,e]-1/(cal['beta']*expected)
                new[z,e]=np.minimum(np.maximum(desired,0),upper[z,e])
        last_error=float(np.max(abs(new-table)))
        table=cfg['individual_damping']*new+(1-cfg['individual_damping'])*table
        for key,value in policy.audit.items():counts[key]=counts.get(key,0)+value
        if progress and (iteration==1 or iteration%100==0):progress(iteration,last_error)
        if last_error<=cfg['individual_tolerance']:
            return PolicyTable(cal,assets,capitals,table,coefficients,cfg['consumption_floor']),dict(
                iterations=iteration,fixed_point_max_error=last_error,interpolation_audit=counts)
    raise RuntimeError(f'individual iteration did not converge: {last_error}; no policy certified')


def generate_shocks(cal,seed,periods,agents):
    rng=np.random.default_rng(np.random.SeedSequence([seed,201001]))
    P=np.asarray(cal['joint_transition']);z=np.zeros(periods,dtype=np.int8)
    employment=np.empty((periods,agents),dtype=np.int8)
    employment[0]=(rng.random(agents)>=cal['unemployment_rates'][0]).astype(np.int8)
    for t in range(1,periods):
        z[t]=int(rng.random()>=P[2*z[t-1],:2].sum())
        rows=2*z[t-1]+employment[t-1]
        p0=P[rows,2*z[t]];p1=P[rows,2*z[t]+1]
        employment[t]=(rng.random(agents)>=p0/(p0+p1)).astype(np.int8)
    return z,employment


def sample_official_panel(mu,edges,employment,seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,201002]));assets=np.empty(len(employment))
    for e in (0,1):
        ids=np.flatnonzero(employment==e)
        choices=rng.choice(len(edges),size=len(ids),p=mu[e]/mu[e].sum())
        positive=choices>0
        values=np.zeros(len(ids));jj=choices[positive]
        values[positive]=edges[jj-1]+rng.random(len(jj))*(edges[jj]-edges[jj-1])
        assets[ids]=values
    return assets


def simulate_panel(policy,initial,z,employment):
    capital=initial.copy();K=np.empty(len(z));max_saving=0.
    for t,zz in enumerate(z):
        K[t]=capital.mean()
        capital=policy.query(K[t],int(zz),capital,employment[t]);max_saving=max(max_saving,float(capital.max()))
    return K,capital,dict(max_individual_saving=max_saving,interpolation_audit=dict(policy.audit))


def fit_alm(K,z,burn):
    B=[];statistics=[]
    for zz in (0,1):
        indices=np.arange(burn,len(K)-1);indices=indices[z[indices]==zz]
        X=np.column_stack((np.ones(len(indices)),np.log(K[indices])));y=np.log(K[indices+1])
        if len(indices)<3 or np.linalg.matrix_rank(X)<2:raise ValueError('rank-deficient ALM regression')
        b=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@b;den=float(np.sum((y-y.mean())**2))
        B.append(b);statistics.append(dict(z=zz,n=len(indices),R2=1-float(res@res)/den if den>0 else None,
                                          regression_RMSE=float(np.sqrt(np.mean(res**2)))))
    return np.asarray(B),statistics
