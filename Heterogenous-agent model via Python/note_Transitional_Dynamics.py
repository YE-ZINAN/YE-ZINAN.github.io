import numpy as np
import numba
from scipy import optimize
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import time

start_time = time.time()
# some useful plot defaults
plt.rcParams.update({'font.size' : 20, 'lines.linewidth' : 3.5, 'figure.figsize' : (13,7)})

#Krusell Smith with aggregate variable in value function
amin,amax,n = -2, 3000, 201
k_space = np.linspace(10,500,25)
sin = 0.25
#income(w,b,tau)
#array([[0.5],
#       [1. ]])
eta=2
beta=0.995
tol=1E-7
alpha=0.36
delta=0.005
tau = 0
b = 0.00000001
N = 0.5/(1.5-0.9565)

#1. Initialize the distribution function of assets F0 and mean K0
#Conjecture the grid points uniformly distributed on [-2,300]
a_space = np.linspace(amin,amax,n)

#Number of grid's value less than 300:
n_300 = 0
for _ in range(len(a_space)):
    if a_space[_] < 300:
        n_300 = _

#Initial distribution
shape = np.zeros((len(k_space),2,n)) #state: (K,e,a)
f_init = np.zeros((2,n))
for e in range(2):
    for a in range(n):
        if a <= n_300:
            f_init[e,a] = 1/2/(n_300+1)
        else:
            f_init[e,a] = 0

#2. Initial mean K:
K = 0
for e in range(2):
    K += np.vdot(f_init[e,:],a_space)

K_init = K
#3. Initialize law of motion of distribution dynamics
def H(K,y0,y1):
    return np.exp(y0 + y1 * np.log(K))
y0 = 0.05
y1 = 0.99

#4. Backward iteration for value fn and policy fn
def Transition_metrics(p,q): 
    return np.array([[p,1-p],[q,1-q]])

def Stationary_distribution(p,q):
    n=Transition_metrics(p,q).shape[0]
    #initial guess of stationary distribution: uniform
    pi = np.full(n,1/n) #pi = [0.5,0.5]
    #Iterate untill stationary
    for i in range(1000):
        pi_next = Transition_metrics(p,q).T @ pi
        if np.max(np.abs(pi_next-pi)) < 1E-10:
            return pi
        else:
            pi = pi_next
            continue
        
p,q = 0.5, 0.0435 
Pi = Transition_metrics(p,q)
pi = Stationary_distribution(p,q)

def rfn(N,K,alpha,delta):
    return alpha * (N/K)**(1-alpha) - delta

def wfn(N,K,alpha):
    return (1-alpha) * (K/N)**alpha

def income(w,b,tau,N,K,alpha):
    w = wfn(N,K,alpha)
    return np.array([[b],[(1-tau)*w]]) #[2,1] metrix, [e=u,e=e]

def backward_iteration(Va,Pi,b,tau,K,beta,eta,N,delta,y0,y1):
    Va1 = np.empty_like(Va)
    for kt in range(len(k_space)):
        k1 = H(k_space[kt],y0,y1)
        w, k = interp_K(k1)
        #print(kt,k,w)
        #Va1[kt] = w * Pi @ Va[k] + (1-w)* Pi @ Va[k+1] #Expectation on Va next period with inital Va
        Va1[kt] = w * Va[k] + (1-w)* Va[k+1]
        Va1[kt] = Pi @ Va1[kt]
    c_endog = (beta * Va1)**(-1/eta) #endogenous c with initial Va
    coh = np.empty_like(Va)
    #Generating a' by interpolating c_endog+a' on cash one hand
    a1 = np.empty_like(Va)
    for k in range(len(k_space)):
        w = wfn(N,k_space[k],alpha)
        r = rfn(N,k_space[k],alpha,delta)
        coh[k] = income(w,b,tau,N,k_space[k],alpha) + (1+(1-tau)*r) * a_space #cash on hand
    for k in range(len(k_space)):
        for e in range(2):
            a1[k,e,:]=np.interp(coh[k,e,:], c_endog[k,e,:] + a_space, a_space) #linear interp
    a1 = np.maximum(a1, a_space[0]) #limitation on borrowing
    c = coh - a1
    Va_new = np.empty_like(Va)
    for k in range(len(k_space)):
        r = rfn(N,k_space[k],alpha,delta)
        Va_new[k] = (1+(1-tau)*r) * c[k]**(-eta) #envelope condition
    return Va_new, a1, c

def interp_K(K):
    K1 = H(K,y0,y1)
    K1 = np.maximum(K1,k_space[0])
    K1 = np.minimum(K1,k_space[-1])
    for k in range(len(k_space)):
        if K1 >= k_space[k] and K1 <= k_space[k+1]:
            interval = [k,k+1]
            break
        else:
            continue
    weight = (k_space[interval[1]] - K1)/(k_space[interval[1]] - k_space[interval[0]])
    return weight, k
    
def steady_state_policy(Pi,a_space,b,tau,eta,beta,tol,K,N,delta,k_space,Hs,y0,y1):
    #initial guess of Va_init: assuming consume 5% of cash on hand
    Va = np.zeros_like(shape) #(nk,2,na)
    a = np.zeros_like(shape)
    c = np.zeros_like(shape)
    a_old = np.zeros_like(shape)
    coh_guess = np.zeros_like(shape)
    c_guess = np.zeros_like(shape)
    Va_guess = np.zeros_like(shape)
    for k in range(len(k_space)):
        w = wfn(N,k_space[k],alpha)
        r = rfn(N,k_space[k],alpha,delta)
        coh_guess[k] = income(w,b,tau,N,K,alpha) + (1+(1-tau)*r) * a_space
        c_guess[k] = 0.05 * coh_guess[k]
        Va_guess[k] = (1+(1-tau)*r) * c_guess[k]**(-eta)
        Va[k] = Va_guess[k]
    k1 = k_space[2]
        #Iteration until error of new and old asset below tolerance:
    for _ in range(10000): #Maximum steps of iteration: 10000
        #print(_)
        Va, a, c = backward_iteration(Va,Pi,b,tau,k1,beta,eta,N,delta,y0,y1)
        if _>0 and np.max(np.abs(a - a_old)) < tol:
            #print(_)
            break
        else:
            a_old = a
            continue
    return Va, a, c

#Policy function and value function
Hs=0
Va ,a, c = steady_state_policy(Pi,a_space,b,tau,eta,beta,tol,K,N,delta,k_space,Hs,y0,y1)     

#5. Simulate the dynamics of the distribution
@numba.njit
def forward_iteration(a, K, wk, kt, a_space, f0):
    f1 = np.zeros_like(f0)

    for k in [kt,kt+1]:
        for e in range(2):
            for i in range(len(a_space)):
                a[k,e,i] = np.maximum(a[k,e,i],a_space[0])
                a[k,e,i] = np.minimum(a[k,e,i],a_space[-1])
                #Search for the interval of household [e,i]'s asset in the asset grid space
                for n in range(len(a_space)):
                    if a[k,e,i] >= a_space[n] and a[k,e,i] <= a_space[n+1]:
                        interval = [n,n+1]
                        break
                    else:
                        continue
                if k == kt:
                    wa = (a_space[interval[1]] - a[k,e,i])/(a_space[interval[1]] - a_space[interval[0]])
                    f1[e,interval[0]] += wk * wa * f0[e,i]
                    f1[e,interval[1]] += wk * (1-wa) * f0[e,i]
                else:
                    wa = (a_space[interval[1]] - a[k,e,i])/(a_space[interval[1]] - a_space[interval[0]])
                    f1[e,interval[0]] += (1-wk) * wa * f0[e,i]
                    f1[e,interval[1]] += (1-wk) * (1-wa) * f0[e,i]
    return Pi.T @ f1

#Given initial K = 148.1, lies between first and second grid of k_space
#Generate the path of distribution 
T = 2000 #Simulation period
flist=[f_init] #flist stores the distribution along simulation path
#Initiate distribution
elist=[]
K_path =[K]

    
def plot_flist():
    plt.plot(a_space[:70],sum(flist[0])[:70],label='Initial')
    plt.plot(a_space[:70],sum(flist[10])[:70],label='Step 10')
    plt.plot(a_space[:70],sum(flist[100])[:70],label='Step 100')
    plt.plot(a_space[:70],sum(flist[500])[:70],label='Step 500')
    plt.plot(a_space[:70],sum(flist[1000])[:70],label='Step 1000')
    plt.plot(a_space[:70],sum(flist[1999])[:70],label='Step 2000')
    plt.xlabel('Individual asset a')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
def plot_elist():
    plt.plot(elist,label='distribution error')
    plt.show()

def plot_kpath():
    plt.plot(K_path,label='K')
    plt.show()

def plot_saving():
    plt.plot(a_space[:100],a[3,1,:][:100] - a_space[:100] ,label='Employed,K=260')
    plt.plot(a_space[:100],a[5,1,:][:100] - a_space[:100] ,label='Employed,K=340')
    plt.plot(a_space[:100],a[3,0,:][:100] - a_space[:100] ,label='Unemployed,K=260')
    plt.legend()
    plt.xlabel('Individual wealth')
    plt.ylabel('Saving a\' - a')
    plt.show()

def plot_va():
    plt.plot(a_space[1:100],Va[3,1,:][1:100] ,label='Employed,K=260')
    plt.plot(a_space[1:100],Va[5,1,:][1:100] ,label='Employed,K=340')
    plt.plot(a_space[1:100],Va[0,1,:][1:100] ,label='Employed,K=140')
    plt.legend()
    plt.xlabel('Individual wealth')
    plt.ylabel('Value')
    plt.show()
    
#6. Use the path of distribution to estimate the law of motion H
#Use first 300 periods to estimate the law of motion
def ols(y,x):
    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    results = model.fit()
    r2 = results.rsquared
    coef = results.params
    return r2, coef

#r2, coef = ols(np.log(K_path[1:1001]),np.log(K_path[0:1000]))

#7/8. Iterate until H converge and test the goodness of fit

coef_list=[[y0,y1]] #list of y0 and y1
ielist=[]
Hs=1
T = 3500
window = 1500
for i in range(1000):
    print('H converge step '+str(i)+', coef:',coef_list[i][0],coef_list[i][1])
    Va ,a, c = steady_state_policy(Pi,a_space,b,tau,eta,beta,tol,K,N,delta,k_space,Hs,y0,y1)     
    K_path = [K_init]
    flist=[f_init]
    K_mean = K_init
    f = f_init
    for _ in range(1,T):
        wk ,kt = interp_K(K_mean)
        f = forward_iteration(a, K_mean,wk ,kt, a_space, f)
        flist.append(f)
        elist.append(np.max(np.abs(flist[-1]-flist[-2])))
        K_mean = 0
        for e in range(2):
            K_mean += np.vdot(f[e],a_space)
        K_path.append(K_mean)
    flist = np.array(flist)
    r2, coef = ols(np.log(K_path[2:2+window]),np.log(K_path[1:1+window]))
    v=0.7
    print('r2:')
    print(r2)
    print('error:')
    y0 = v*coef[0] + (1-v)*coef_list[i][0]
    y1 = v*coef[1] + (1-v)*coef_list[i][1]
    coef_list.append(coef)
    print(np.max(np.abs(coef_list[-1]-coef_list[-2])))
    print('\n')
    ielist.append(K_path[-1])
    if np.max(np.abs(coef_list[-1]-coef_list[-2])) < 1E-5:
        print('H converged at step '+str(i))
        print('R-squared of final step: '+str(r2))
        print('beta0 and beta1: '+str(coef[0])+', '+str(coef[1]))
        end_time = time.time()
        runtime = end_time - start_time
        hours = int(runtime / 3600)
        minutes = int((runtime % 3600) / 60)
        seconds = int(runtime % 60)
        print(f"Run time：{hours} h, {minutes} m, {seconds} s with v = "+str(v))
        plot_kpath()
        break
    else:
        continue




