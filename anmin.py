import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import TensorDataset
from time import time

def Fmat(A,x):
    x1=Fn.relu(x@A)
    u=(x1>0).float()
    return u

def anet1(A,b,b0,F,x):
    Ax=x@A
    py=(F*Ax)@b+b0
    return py

def anet(A,b,b0,x,al):
    xa=x@A
    Ax=Fn.relu(xa)*(1-al)+xa*al
    py=Ax@b+b0
    return py

def fitOLSw(x,y,w,la=0.0001):
    x1=torch.cat((torch.ones(x.shape[0],1,device=x.device),x),axis=1)
    xx=x1.t()@(x1*w.view(-1,1))
    xy=x1.t()@(y*w.view(-1,1))
    b=torch.linalg.solve(xx+torch.eye(xx.shape[0],device=x.device)*la,xy)
    return b[1:],b[0]

def fitAw(F,x,y,w,b,b0,la=0.0001):
    # find A to minimize w||U*(xA)b+b0-y||^2
    data = TensorDataset(x,F,w,y.view(-1))
    loader = torch.utils.data.DataLoader(data, batch_size=2048*4, shuffle=True)
    yb0=y-b0
    xy=w.view(-1,1)*yb0*x
    uxy=F.t()@xy
    buxy=uxy*b
    buxy=buxy.reshape(-1,1)
    d=x.shape[1]
    h=F.shape[1]
    dh=d*h
    xx=torch.zeros(h,h,d,d,device=x.device)
    for xi,ui,wi,yi in loader:
        xu=xi.t().unsqueeze(0)*ui.t().unsqueeze(1)
        #print(xu.shape,wi.shape)
        xxi=(xu*wi.view(1,1,-1)).unsqueeze(1)@xu.permute(0,2,1).unsqueeze(0)
        xx+=xxi
    xxb=xx*(b@b.t()).unsqueeze(2).unsqueeze(3)
    xxr=xxb.permute(0,2,1,3).reshape(dh,dh)
    xxr1=xxr+torch.eye(dh,device=x.device)*la
    ld=torch.logdet(xxr1)
    #print('logdet=%1.2f',ld)
    if ld>-10000:
        A=torch.linalg.solve(xxr1,buxy)
    else:
        u,d,vt=torch.linalg.svd(xxr1)
        d[d<0.0001]=0.0001
        xi=u@torch.diag(d)@vt
        A=xi@buxy
    A=A.reshape(h,d).t()
    return A

def fitOLS(x,y,la=0.0001):
    x1=torch.cat((torch.ones(x.shape[0],1,device=x.device),x),axis=1)
    xx=x1.t()@x1
    xy=x1.t()@y
    b=torch.linalg.solve(xx+torch.eye(xx.shape[0],device=x.device)*la,xy)
    if len(b.shape)==1:
        return b[1:],b[0]
    else:
        return b[1:,:],b[0,:]

def fitA(F,x,y,b,b0,la=0.0001):
    # find A to minimize ||U*(xA)b+b0-y||^2
    #ny=y.shape[1]
    data = TensorDataset(x,F,y)
    loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True)
    yb0=y-b0
    if len(b.shape)==1:
        ybb=yb0.view(-1,1)*b.view(1,-1)
        ybbf=ybb*F
        buxy=ybbf.t()@x
        buxy=buxy.reshape(-1,1)
    else:
        ybb=yb0@b.t()
        bxy=ybb.unsqueeze(1)*x.unsqueeze(2)
        buxy=torch.sum(F.unsqueeze(1)*bxy,dim=0)
#    print(U.shape,buxy.shape)
        buxy=buxy.t().reshape(-1,1)
    t1=time()
    d=x.shape[1]
    h=F.shape[1]
    n=d*h
    xx=torch.zeros(h,h,d,d,device=x.device)
#    if len(b.shape)==1:
#        for xi,fi,yi in loader:
#            xxi=xxi.permute(1,2,0).unsqueeze(0)
#            uui=uui.permute(2,0,1).unsqueeze(1)
#            xxi1=xxi@uui
            #print(xxi1.shape)
            #xu=xi.unsqueeze(1)*fi.unsqueeze(2)
            #xxi=xu.permute(1,2,0).unsqueeze(1)@xu.permute(1,0,2).unsqueeze(0)
#    else:
    for xi,fi,yi in loader:
        #print(xxi.shape,uui.shape)
        xu=xi.t().unsqueeze(0)*fi.t().unsqueeze(1)
        xxi=xu.unsqueeze(1)@xu.permute(0,2,1).unsqueeze(0)
        xx+=xxi
    t2=time()
    #print(xu.shape,xx1.shape)   
    #for i in range(h):
    #    xxi=xu[i,:,:].unsqueeze(0)@xu.permute(0,2,1)
    #    xx[i,:,:,:]=xxi
    #print(torch.min(xx-xx1),torch.max(xx-xx1))
    if len(b.shape)==1:
        xxb=xx*(b.view(-1,1)*b.view(1,-1)).unsqueeze(2).unsqueeze(3)
    else:
        xxb=xx*(b@b.t()).unsqueeze(2).unsqueeze(3)
    xxr=xxb.permute(0,2,1,3).reshape(n,n)
    xxr1=xxr+torch.eye(n,device=x.device)*la
    ld=torch.logdet(xxr1).item()
    t3=time()
    #print(ld)
    if ld>-10000:
        A=torch.linalg.solve(xxr1,buxy)
    else:
        #A=torch.randn((d,h),device=x.device)
        u,D,vt=torch.linalg.svd(xxr1)
        D[D<0.0001]=0.0001
        xi=u@torch.diag(1/D)@vt
        A=xi@buxy
    A=A.reshape(h,d).t()
    t4=time()
    print(t2-t1,t3-t2,t4-t3)
    return A,ld
