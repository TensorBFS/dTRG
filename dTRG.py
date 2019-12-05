import os
import sys
import re
import torch
import time
import numpy as np
from utils import kronecker_product as kron
from torch.utils.checkpoint import checkpoint
from datetime import datetime
from expm import expm

class iXTRG(torch.nn.Module):
    def __init__(self,args,NBeta,dtype,device):
        super(iXTRG, self).__init__()

        # model
        self.NBeta = NBeta  # no. of isometries on different layers, starts from 1
        self.tau = torch.tensor([args.tau], dtype=dtype, device=device)
        self.model = args.model

        # algorithm
        self.D = args.D 
        self.opti = args.opti
        self.Niter = args.Niter
        self.Nsweep = args.Nsweep
        self.depth = args.depth

        # io
        self.dtype = dtype
        self.device = device
        self.rdir = args.rdir
        
        self.fName = self.rdir+ 'Fe_'+self.model+'_tau'+str(self.tau.item())+'_NBeta'+str(self.NBeta)+\
                     '_Dc'+str(self.D)+'_opti'+str(self.opti)+\
                     '_Niter'+str(self.Niter)+'_Nswp'+str(self.Nsweep)+\
                     '_depth'+str(self.depth)+'_device'+str(args.cuda)+'.pt'
       
        # create dir if not existed
        if os.path.isdir(self.rdir) is False: os.makedirs(self.rdir)

        # Module Parameters
        """
        [Wl_1, Wl_2, ..., Wl_NBeta, Wr_1, Wr_2, ..., Wr_NBeta]
        """
        """
        2nd-order Trotter-Suzuki Decomposition!!!
        """
        nb = self.NBeta-1
        Dl = min(4**(nb+2), self.D)
        Dr = min(4**(nb+2), self.D)
        D2 = min(4**(nb+3), self.D)
        Wl.append(torch.randn(Dl,Dr,D2,dtype=dtype,device='cpu'))
            
        Dl = min(4**(nb+1), self.D)
        Dr = min(4**(nb+1), self.D)
        D2 = min(4**(nb+2), self.D)
        Wr.append(torch.randn(Dl,Dr,D2,dtype=dtype,device='cpu'))
        
        self.params = torch.nn.ParameterList(
                        [torch.nn.Parameter(_.to(device)) for _ in Wl+Wr]
                        )
        self.params.append(torch.nn.Parameter(self.tau))
         
    def turn_on_grad(self,i):
        for j, p in enumerate(self.parameters()):
            if (j==i): p.requires_grad = True
            else:      p.requires_grad = False

    def getHamilton(self):
        # spin operator 
        sz = torch.tensor([[1, 0],[0,-1]], dtype=dtype, device=device)/2
        sx = torch.tensor([[0, 1],[1, 0]], dtype=dtype, device=device)/2
        sy = torch.tensor([[0,-1],[1, 0]], dtype=dtype, device=device)/2

        if self.model == "Ising":
            H = kron(sz,sz)
        elif self.model == "XY":
            H = kron(sx,sx) - kron(sy,sy)
        return H

    def getisometry(self, step, sizeT):
        D = min(self.D, sizeT)
        D_new = min(self.D, D**2)
        return self.params[step][:D,:D,:D_new]

    def getMaxEigBiLayer(self, Ta, Tb):
        # initial boundary 'vector'
        Vl = torch.einsum('ijkl,mjkl->im',Ta,Ta)
        normFactor = torch.norm(Vl)
        Vl = Vl/normFactor

        # Power method
        matchCnt = 0
        for _ in range(400):
            Vl = torch.einsum('im,ijkl,mnkl->jn',Vl,Ta,Ta)
            Vl = torch.einsum('im,ijkl,mnkl->jn',Vl,Tb,Tb)
            if (torch.norm(Vl)-normFactor)/normFactor<1e-8: matchCnt +=1 
            if matchCnt==5: break
            normFactor = torch.norm(Vl)
            Vl = Vl/normFactor
            if _==399: print('Eig not well converged!',end=' ')
        return normFactor


    def initRho(self, trotter_order=2):
        tau,device,dtype,model = self.params[-1],self.device,self.dtype,self.model
        #print("Generate initial rho({}) via Trotter decomp.\n".format(tau))
        if trotter_order==1:
            # get Hamiltonian
            H = self.getHamilton()
            # local trotter gate
            rho = expm(-tau*H).view(2,2,2,2)
            rho = torch.einsum('ijkl->ikjl',rho).contiguous().view(4,4)
            # svd & truncate the 0 values
            U,S,V = torch.svd(rho)
            # trotter gate in form of two tensor contraction
            hl = (U@torch.diag(torch.sqrt(S))).view(2,2,4)
            hr = (V@torch.diag(torch.sqrt(S))).view(2,2,4)
            # local tensor of initial mpo, index order: [l,r,d,u] 
            Ta = torch.tensordot(hr,hl,([0],[1])).permute(1,3,2,0).contiguous()
            Tb = torch.tensordot(hr,hl,([1],[0])).permute(1,3,0,2).contiguous()
        elif trotter_order==2:
            # get Hamiltonian
            H = self.getHamilton()
            # local trotter gate
            rho = expm(-tau*H).view(2,2,2,2)
            rho = torch.einsum('ijkl->ikjl',rho).contiguous().view(4,4)
            # half local trotter gate
            rho_half = expm(-tau*H/2).view(2,2,2,2)
            rho_half = torch.einsum('ijkl->ikjl',rho_half).contiguous().view(4,4)
            # svd
            U,S,V = torch.svd(rho)
            U2,S2,V2 = torch.svd(rho_half)
            # trotter gate in form of two tensor contraction
            hl = (U@torch.diag(torch.sqrt(S))).view(2,2,4)
            hr = (V@torch.diag(torch.sqrt(S))).view(2,2,4)
            hl2 = (U2@torch.diag(torch.sqrt(S2))).view(2,2,4)
            hr2 = (V2@torch.diag(torch.sqrt(S2))).view(2,2,4)
            # local tensor of initial mpo, index order: [l,r,d,u] 
            Ta = torch.einsum('ijk,jlm,lno->okmin',hr2,hl,hr2).contiguous().view(16,4,2,2)
            Tb = torch.einsum('ijk,jlm,lno->mokin',hl2,hr,hl2).contiguous().view(4,16,2,2)
        else:
            raise Exception('only 1st and 2nd trotter are available!')
        return Ta,Tb 


    def forward(self, nlayer):
        if nlayer == 0:
            [Ta, Tb] = self.initRho()
            lnZ = 0.0
        else:
            Ta = Tas[nlayer-1].to(device)
            Tb = Tbs[nlayer-1].to(device)
            lnZ = lnZs[nlayer-1].to(device)

        for nbeta in range(nlayer,self.NBeta):
           Sizea = list(Ta.size())
           Sizeb = list(Tb.size())

           # obtain Isometry tensor from antisymmetric tensor
           Wa = self.getisometry(nbeta,Sizea[0])
           Wb = self.getisometry(nbeta+self.NBeta,Sizeb[0])

           # evolution Ta & Tb
           Ta = checkpoint(self.rgtens,Wa,Ta,Wb)
           Tb = checkpoint(self.rgtens,Wb,Tb,Wa)
           lnZ = 2*lnZ + torch.log(torch.norm(Ta))\
                       + torch.log(torch.norm(Tb))
           Ta = Ta/torch.norm(Ta)
           Tb = Tb/torch.norm(Tb)

           if len(Tas)<nbeta+1: Tas.append(Ta.detach().to('cpu'))
           else:                Tas[nbeta]=Ta.detach().to('cpu')
           if len(Tbs)<nbeta+1: Tbs.append(Tb.detach().to('cpu'))
           else:                Tbs[nbeta]=Tb.detach().to('cpu')
           if len(lnZs)<nbeta+1: lnZs.append(lnZ.detach().to('cpu'))
           else:                 lnZs[nbeta]=lnZ.detach().to('cpu')

        # free energy
        ee = self.getMaxEigBiLayer(Ta,Tb)
        lnZ = 1/2*(2*lnZ + torch.log(ee))
        return lnZ

    def rgtens(self, Wa, T, Wb):
        T = torch.einsum('ijk,ilmn,jopm,loq->kqpn',Wa,T,T,Wb)
        return T

    def update_single_layer(self, nlayer):
        loss_old = 0
        for niter in range(self.Niter):
            for ii in range(2):
                print('\t\tW(%02d-%02d),'%(nlayer,ii),end=' ')
                self.turn_on_grad(nlayer+self.NBeta*ii)
        
                self.zero_grad()
                loss = self.forward(nlayer)
                loss.backward()
                
                with torch.no_grad():
                   E = self.params[nlayer+self.NBeta*ii].grad
                   D, D, D_new = E.shape
                   E = E.view(D**2, D_new)
                   # perform MERA update
                   U,S,V = torch.svd(E)
                   self.params[nlayer+self.NBeta*ii].data = (U@V.t()).view(D,D,D_new)       
                   print('%+.15f'%(-loss.item()/2**(self.NBeta+1)/self.tau), end=' ')
            if abs((loss_old - loss.item())/loss.item())<1e-8: break
            else: 
                loss_old = loss.item()
            print(' ')
        return loss.item()
        

"""
===================================================================================
Main entry of the program
===================================================================================
"""
if __name__=="__main__":
    from args import args
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))
    dtype = torch.float32 if args.use_float32 else torch.float64
    FeS = []  # init thermal quantities

    Wl = []; Wr = []
    Tas = []; Tbs = []; lnZs = []
    for nbeta in range(args.NBeta):  # main loop, cooling down the system
        beta = 2**(nbeta+1)*args.tau 
        model = iXTRG(args, nbeta+1, dtype, device)    
        print('\n '+'-'*40)
        print('Optimizing lnZ(%.5f)'%(2*beta))
        print('\nInitialization:')
        loss = model.update_single_layer(nbeta)
        
        opti_flag = 0   # no sweep by default
        if(nbeta+1 >= model.opti) :
           opti_flag = 1

        depth = min(model.depth,nbeta)
        # call sweep optimization
        if opti_flag:
            print('\noptimiztion:')
            for nswp in range(args.Nsweep):
                print('#sweep {}:'.format(nswp))
                # nlayer is No. of layer to be optimized, from 1 to 
                for nlayer in range(nbeta-depth,nbeta+1):
                    loss = model.update_single_layer(nlayer) 
                    print(' ')
        
        # saving isometric tensor for next iteration
        Wl = [m.detach().to('cpu') for m in model.params[:nbeta+1]]
        Wr = [m.detach().to('cpu') for m in model.params[nbeta+1:-1]]
        
        # remove redundant isometries (to save memory)
        if (nbeta-depth >= 1):
           Wl[nbeta-depth-1] = torch.tensor([0.])
           Wr[nbeta-depth-1] = torch.tensor([0.])
        
        del model.params
        torch.cuda.empty_cache()
         
    # save data    
    torch.save(FeS, model.fName)
