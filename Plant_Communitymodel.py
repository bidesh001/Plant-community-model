"""

Code to solve the plant community model

We use a pseudo-spectral method. Time integration was performed using a 
fourth order Runge Kutta scheme.

@authors: Bidesh K. Bera, Omer Tzuk, Jamie J. R. Bennett, Ehud Meron
"""
import numpy as np
#%% Parameters values
lam=0.032 # growth rate (\lambda_0)
gam=20.0 # water uptake rate (\Gamma)
ns=4.0 # evaporation rate in bare soil (L_0)
rr=10.0 # evaporation reduction due to shading (R)
ff=0.01 # infiltration contrast (f)
qq=0.06 # reference biomass (Q)
aa=40.0 # maximum value of infiltration rate (A)
p=150.0 # precipitation rate (P)
kmin=0.1 # minimal capacity to capture light (K_min)
kmax=0.6 # maximal capacity to capture light (K_max)
mmin=0.5 # minimal mortality rate (M_min)
mmax=0.9 # maximal mortality rate (M_max)
ymin=0.5 # minimal contribution to infiltration rate (Y_min)
ymax=1.5 # maximal contribution to infiltration rate (Y_max)

deltab=1.0 # biomass dispersal rate (D_B)
deltaw=100.0 # soil-water diffusion coefficient (D_W)
deltah=10000.0 # overland-water diffusion coefficient (D_H) 
deltachi=0.000001 # trait diffusion rate (D_\chi)

dt=0.01 # step size
NNX=128*8 # number of spatial grid points
NNC=128 # number of functional groups (N)
domainsize=800 # length of spatial domain
tmax=4000 # end time
steps=int(tmax/dt) # time step
dchi=deltachi/(1/NNC)**2 
#%%
Lx=domainsize/2
Lchi=1/2
x=(2*Lx/NNX)*np.arange(-NNX/2,NNX/2,1)
chim=np.linspace(0,1,NNC) # tradeoff parameter (\chi)
kchim=kmax+(chim)*(kmin-kmax) # tradeoff relation through K
mchim=mmax+(chim)*(mmin-mmax) # tradeoff relation through M
ychim=ymax+(chim)*(ymin-ymax) # tradeoff relation through Y

kchim=np.reshape(kchim,(NNC,1))
mchim=np.reshape(mchim,(NNC,1))
real_space=np.linspace(0,domainsize,NNX)
trait_space=np.linspace(0,1,NNC)
#%%
f = lambda b,w,s1: lam*(kchim/(s1+kchim))*w*b-mchim*b
g = lambda w,h,s1,s2: aa*((s2+ff*qq)/(s2+qq))*h-((ns*w)/(1.0+rr*s1))-gam*w*s1
k = lambda h,s2: p-aa*((s2+ff*qq)/(s2+qq))*h
#Initialise
b = np.ones([NNC,NNX])*0.12 + np.random.uniform(0,0.001,(NNC,NNX))
w = np.ones([NNX])*0.72+np.random.uniform(0,0.01,(NNX))*0.5
h = np.ones([NNX])*0.83+np.random.uniform(0,0.01,(NNX))*0.5
#%%
bhat=np.fft.fft(b,axis=1)
what=np.fft.fft(w)
hhat=np.fft.fft(h)
#%%
BB=b/NNC
s=np.empty([NNC,NNX]) 
storeb=np.array([BB])

kx=(np.pi/Lx)*np.concatenate((np.arange(0,NNX/2 +1),np.arange(-NNX/2 +1,0)))
kchi=(np.pi/Lchi)*np.concatenate((np.arange(0,NNC/2 +1),np.arange(-NNC/2 +1,0)))
[kxx,kchichi]=np.meshgrid(kx,kchi)

ksqb=(deltab*kxx**2)
ksqw=(deltaw*kxx**2)
ksqh=(deltah*kxx**2)
 
print_every = 0.1*steps

Eb=np.exp(-1.0*dt*ksqb/2)
Eb2=Eb**2
Ew=np.exp(-1.0*dt*ksqw/2)
Ew2 = Ew**2
Eh=np.exp(-1.0*dt*ksqh/2)
Eh2 = Eh**2
#%% Runge–Kutta Method
for n in range(1,steps):    
    s[0,]=(b[0,]+b[1,]-2.0*b[0,])*dchi
    s[1:NNC-1,]=(b[0:NNC-2,]+b[2:NNC,]-2.0*b[1:NNC-1,])*dchi
    s[NNC-1,]=(b[NNC-2,]+b[NNC-1,]-2.0*b[NNC-1,])*dchi
       
    sn=np.sum(b, axis = 0)
    sy=np.sum(np.reshape(ychim,(NNC,1))*b,axis=0)
    s1=sn/NNC
    s2=sy/NNC
       
    dbi, dwi, dhi = f(b,w,s1)+s, g(w,h,s1,s2), k(h,s2)
    k1b, k1w, k1h = dt*np.fft.fft(dbi,axis = 1), dt*np.fft.fft(dwi), dt*np.fft.fft(dhi)
    b2=np.real(np.fft.ifft(Eb*(bhat+k1b/2),axis = 1))
    w2=np.real(np.fft.ifft(Ew*(what+k1w/2)))
    h2=np.real(np.fft.ifft(Eh*(hhat+k1h/2)))
        
    s[0,]=(b2[0,]+b2[1,]-2.0*b2[0,])*dchi
    s[1:NNC-1,]=(b2[0:NNC-2,]+b2[2:NNC,]-2.0*b2[1:NNC-1,])*dchi
    s[NNC-1,]=(b2[NNC-2,]+b2[NNC-1,]-2.0*b2[NNC-1,])*dchi
        
    sn=np.sum(b2, axis = 0)
    sy=np.sum(np.reshape(ychim,(NNC,1))*b2, axis = 0)
    s1=sn/NNC
    s2=sy/NNC
        
    dbi,dwi,dhi = f(b2,w2,s1)+s, g(w2,h2,s1,s2), k(h2,s2)
    k2b, k2w, k2h = dt*np.fft.fft(dbi,axis = 1), dt*np.fft.fft(dwi), dt*np.fft.fft(dhi)
    b3=np.real(np.fft.ifft(Eb*bhat+k2b/2,axis = 1)) 
    w3=np.real(np.fft.ifft(Ew*what+k2w/2))
    h3=np.real(np.fft.ifft(Eh*hhat+k2h/2))
       
    s[0,]=(b3[0,]+b3[1,]-2.0*b3[0,])*dchi
    s[1:NNC-1,]=(b3[0:NNC-2,]+b3[2:NNC,]-2.0*b3[1:NNC-1,])*dchi
    s[NNC-1,]=(b3[NNC-2,]+b3[NNC-1,]-2.0*b3[NNC-1,])*dchi
        
    sn=np.sum(b3, axis = 0)
    sy=np.sum(np.reshape(ychim,(NNC,1))*b3, axis = 0)
    s1=sn/NNC
    s2=sy/NNC
        
    dbi,dwi,dhi = f(b3,w3,s1)+s, g(w3,h3,s1,s2), k(h3,s2)
    k3b, k3w, k3h = dt*np.fft.fft(dbi,axis = 1), dt*np.fft.fft(dwi), dt*np.fft.fft(dhi);
    b4=np.real(np.fft.ifft(Eb2*bhat+Eb*k3b,axis = 1))
    w4=np.real(np.fft.ifft(Ew2*what+Ew*k3w))
    h4=np.real(np.fft.ifft(Eh2*hhat+Eh*k3h))
        
    s[0,]=(b4[0,]+b4[1,]-2.0*b4[0,])*dchi
    s[1:NNC-1,]=(b4[0:NNC-2,]+b4[2:NNC,]-2.0*b4[1:NNC-1,])*dchi
    s[NNC-1,]=(b4[NNC-2,]+b4[NNC-1,]-2.0*b4[NNC-1,])*dchi
        
    sn=np.sum(b4, axis = 0)
    sy=np.sum(np.reshape(ychim,(NNC,1))*b4, axis = 0)
    s1=sn/NNC
    s2=sy/NNC
        
    dbi,dwi,dhi = f(b4,w4,s1)+s, g(w4,h4,s1,s2), k(h4,s2)
    k4b, k4w, k4h = dt*np.fft.fft(dbi,axis = 1), dt*np.fft.fft(dwi), dt*np.fft.fft(dhi)
    bhat=Eb2*bhat+(Eb2*k1b+2*Eb*(k2b+k3b)+k4b)/6
    what=Ew2*what+(Ew2*k1w+2*Ew*(k2w+k3w)+k4w)/6
    hhat=Eh2*hhat+(Eh2*k1h+2*Eh*(k2h+k3h)+k4h)/6

    b=np.real(np.fft.ifft(bhat,axis = 1)); w=np.real(np.fft.ifft(what)); h=np.real(np.fft.ifft(hhat))
    bhat=np.fft.fft(b,axis = 1); what=np.fft.fft(w); hhat=np.fft.fft(h);
    
    BB=b/NNC

    if (n % print_every) == 0:
            print(int(100*n/steps))
storeb=np.append(storeb,[BB],axis=0)
B=storeb[-1] # Biomass in trait and real space
B_avg=np.array([])
for i in range(0,NNC):
    for j in range(0,NNX):
        kj=np.mean(storeb[-1,i,:])
    B_avg=np.append(B_avg,kj) # Spatial averaged biomass in trait space  

deno=np.sum(B_avg)
bs1=B_avg/deno
bs2=bs1*np.log(bs1)
sh=-np.sum(bs2) # Shanon diversity index
pindex=sh/np.log(NNC) # Pielou index
#np.savez('Communitymodel_P_150',biomass=storeb[-1],biomass_avg=B_avg,evenness=pindex) #To save data


         