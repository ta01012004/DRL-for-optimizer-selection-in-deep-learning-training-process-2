# ppo_optimizer_selector.py — v4.1 full
"""
PPO agent chọn Optimizer (SGD / Adam / SAM) cho ResNet‑18 trên CIFAR‑10
=====================================================================
• Mỗi episode = EP_LEN epoch; agent quyết định optimizer từng epoch.
• Reward = Δ validation‑accuracy.
• Lưu & hiển thị 12 metrics (Acc / Prec / F1 / Loss • Train/Val/Test) + Reward + Time.
• Biểu đồ: Reward, Accuracy, Loss, Precision, F1, Time.
• Early‑stopping nếu Val‑Acc không cải thiện PATIENCE_EP episode.
"""

import time, random, math
from dataclasses import dataclass
from typing import List

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torchvision import models
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score

# ---------------- CONFIG -----------------
SEED = 42; random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_TRAIN, BATCH_EVAL = 128, 256
VAL_RATIO = 0.1; NUM_CLASSES = 10
EP_LEN, MAX_EP, PATIENCE_EP = 10, 150, 15
GAMMA, LAMBDA, CLIP = 0.99, 0.95, 0.2
LR, ENT_COEF, VAL_COEF = 3e-4, 0.01, 0.5
MAX_GRAD, PPO_EPOCHS, BATCH_PPO = 0.5, 4, 64
OPT_NAMES = ['sgd', 'adam', 'sam']

# -------------- DATA --------------------

def cifar_load():
    tf_t = T.Compose([T.RandomCrop(32,4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    tf_e = T.Compose([T.ToTensor(),T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    full=CIFAR10('./data',True,download=True,transform=tf_t); test=CIFAR10('./data',False,download=True,transform=tf_e)
    n_val=int(len(full)*VAL_RATIO)
    tr,val=random_split(full,[len(full)-n_val,n_val],generator=torch.Generator().manual_seed(SEED))
    return DataLoader(tr,BATCH_TRAIN,True,2,pin_memory=True),DataLoader(val,BATCH_EVAL,False,2,pin_memory=True),DataLoader(test,BATCH_EVAL,False,2,pin_memory=True)

# -------------- MODEL & SAM ------------

def resnet18_scratch():
    m=models.resnet18(weights=None); m.fc=nn.Linear(m.fc.in_features,NUM_CLASSES); return m

class SAM(torch.optim.Optimizer):
    def __init__(self,params,base,rho=0.05,**kw):
        params=list(params); self.base=base(params,**kw); self.param_groups=self.base.param_groups; self.rho=rho; super().__init__(params,{})
    @torch.no_grad()
    def first_step(self):
        gn=torch.norm(torch.stack([p.grad.norm() for g in self.param_groups for p in g['params'] if p.grad is not None])); s=self.rho/(gn+1e-12)
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None: continue
                e=p.grad*s; p.add_(e); self.state[p]['e']=e
    @torch.no_grad()
    def second_step(self):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e'])
        self.base.step()
    def step(self,closure):
        loss=closure(); self.first_step(); closure(); self.second_step(); return loss

# -------------- ENV --------------------
class Env:
    def __init__(self):
        self.tr,self.val,self.test=cifar_load(); self.loss=nn.CrossEntropyLoss(); self.reset()
    @property
    def obs_dim(self): return 3+len(OPT_NAMES)
    @property
    def act_dim(self): return len(OPT_NAMES)
    def _make_opt(self,name):
        if name=='sgd':  return torch.optim.SGD(self.model.parameters(),lr=1e-2,momentum=0.9)
        if name=='adam': return torch.optim.Adam(self.model.parameters(),lr=1e-3)
        if name=='sam':  return SAM(self.model.parameters(),torch.optim.SGD,lr=1e-2,momentum=0.9)
    def _train(self,opt):
        self.model.train()
        for x,y in self.tr:
            x,y=x.to(DEVICE),y.to(DEVICE)
            if isinstance(opt,SAM):
                def clo(): opt.zero_grad(); o=self.model(x); l=self.loss(o,y); l.backward(); return l
                opt.step(clo)
            else:
                opt.zero_grad(); o=self.model(x); l=self.loss(o,y); l.backward(); opt.step()
    def _eval(self,loader):
        self.model.eval(); ls=0; preds=[]; labs=[]
        with torch.no_grad():
            for x,y in loader:
                x,y=x.to(DEVICE),y.to(DEVICE); o=self.model(x); ls+=self.loss(o,y).item()*x.size(0); preds.append(o.argmax(1).cpu()); labs.append(y.cpu())
        p=np.concatenate(preds); l=np.concatenate(labs)
        acc=(p==l).mean(); prec=precision_score(l,p,average='weighted',zero_division=0); f1=f1_score(l,p,average='weighted',zero_division=0)
        return acc,prec,f1,ls/len(loader.dataset)
    def reset(self):
        self.model=resnet18_scratch().to(DEVICE); a,_,_,l=self._eval(self.val); self.p_acc,self.p_loss=a,l; self.ep=0; self.p_act=1; return self._state()
    def _state(self):
        return torch.tensor([self.p_acc,self.p_loss,self.ep/EP_LEN,*[int(self.p_act==i) for i in range(len(OPT_NAMES))]],dtype=torch.float32,device=DEVICE)
    def step(self,act):
        opt=self._make_opt(OPT_NAMES[act]); t=time.time(); self._train(opt); dur=time.time()-t
        a,pr,f1,l=self._eval(self.val); r=float(a-self.p_acc)
        self.p_acc,self.p_loss=a,l; self.p_act=act; self.ep+=1; d=self.ep>=EP_LEN
        return self._state(),r,d,{'dur':dur,'val_prec':pr,'val_f1':f1}
    def full_metrics(self):
        tr_acc,tr_pr,tr_f1,tr_ls=self._eval(self.tr)
        te_acc,te_pr,te_f1,te_ls=self._eval(self.test)
        return tr_acc,tr_pr,tr_f1,tr_ls,self.p_acc,self._eval(self.val)[1],self._eval(self.val)[2],self.p_loss,te_acc,te_pr,te_f1,te_ls

# -------------- AGENT ------------------
class ActorCritic(nn.Module):
    def __init__(self,obs,act):
        super().__init__(); self.b=nn.Sequential(nn.Linear(obs,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh()); self.pi=nn.Linear(64,act); self.v=nn.Linear(64,1)
    def forward(self,x): h=self.b(x); return self.pi(h),self.v(h)

def gae(rew,vals,dons,next_v):
    adv,gae=[],0; vals=vals+[next_v]
    for t in reversed(range(len(rew))):
        d=rew[t]+GAMMA*vals[t+1]*(1-dons[t])-vals[t]; gae=d+GAMMA*LAMBDA*(1-dons[t])*gae; adv.insert(0,gae)
    ret=[a+v for a,v in zip(adv,vals[:-1])]
    adv=torch.tensor(adv,dtype=torch.float32,device=DEVICE); adv=(adv-adv.mean())/(adv.std()+1e-8)
    return adv,torch.tensor(ret,dtype=torch.float32,device=DEVICE)

@dataclass
class Traj: s:List[torch.Tensor]; a:List[torch.Tensor]; lp:List[torch.Tensor]; r:List[float]; v:List[torch.Tensor]; d:List[bool]

def ppo_update(net,opt,tr:Traj):
    S=torch.stack(tr.s); A=torch.stack(tr.a); old_lp=torch.stack(tr.lp); V=torch.stack(tr.v).squeeze()
    with torch.no_grad(): _,nxt=net(tr.s[-1])
    adv,ret=gae(tr.r,V.tolist(),tr.d,nxt.item())
    for _ in range(PPO_EPOCHS):
        idx=torch.randperm(len(S))
        for st in range(0,len(S),BATCH_PPO):
            b=idx[st:st+BATCH_PPO]; logits,val=net(S[b]); dist=torch.distributions.Categorical(logits=logits)
            lp=dist.log_prob(A[b]); ratio=(lp-old_lp[b]).exp(); s1=ratio*adv[b]; s2=torch.clamp(ratio,1-CLIP,1+CLIP)*adv[b]
            loss_pi=-torch.min(s1,s2).mean(); loss_v=F.mse_loss(val.squeeze(),ret[b]); ent=dist.entropy().mean()
            loss=loss_pi+VAL_COEF*loss_v-ENT
