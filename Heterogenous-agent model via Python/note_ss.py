import numpy as np
import numba
from scipy import optimize
import matplotlib.pyplot as plt

# some useful plot defaults
plt.rcParams.update({'font.size' : 20, 'lines.linewidth' : 3.5, 'figure.figsize' : (13,7)})

#Model:

amin,amax,n = 0, 10000, 100
#Government balance: B=T, T = tau(wN+rK)

#Infinite living households
#Househould utility: u(c)=c^(1-eta) / (1-eta)
#Value: V(e,a) = max u(c) + beta * E[V'(e',a')|e]
#Househoulds'
#①budget constraints:
# a' = (1+(1-tau)*r)*a + (1-tau)w - c  if e = employed
# a' = (1+(1-tau)*r)*a + b - c if e = unemployed
#②asset constraint: a' >= amin


#uninsurable employment status following transition pi(e'|e)
#For AR(1): Rouwenhorse method
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
#Pi=
#[ Pu_u,  Pu_e
#  Pe_u,  Pe_e]
pi = Stationary_distribution(p,q) # pi = array([0.0800368, 0.9199632])

    
#Asset grids
#ai = amin + e^(e^(ui-1)) - 1
def asset_grids(amin,amax,n):
    umin = np.log(1+np.log(amax-amin))
    #Generate initial grids:
    grid = np.linspace(0,umin,n)
    return amin + np.exp(np.exp(grid)-1)-1


a_space = asset_grids(amin,amax,n)
#array([0.00000000e+00, 2.18583203e-02,...1.25698865e+03, 1.49900000e+03])
#len(a_space)=100


#Obtaining steady-state policies: c, a'
def income(w,b,tau):
    return np.array([[b],[(1-tau)*w]]) #[2,1] metrix, [e=u,e=e]

def backward_iteration(Va_init,Pi,w,b,tau,r,beta,eta):
    Va1_init = Pi @ Va_init #Expectation on Va next period with inital Va
    c_endog = (beta * Va1_init)**(-1/eta) #endogenous c with initial Va
    coh = income(w,b,tau) + (1+(1-tau)*r) * a_space #cash on hand
    #Generating a' by interpolating c_endog+a' on cash one hand
    a1 = np.empty_like(coh)
    for e in range(2):
        a1[e,:]=np.interp(coh[e,:], c_endog[e, :] + a_space, a_space) #linear interp
    a1 = np.maximum(a1, a_space[0]) #limitation on borrowing
    c = coh - a1
    Va_new = (1+(1-tau)*r) * c**(-eta) #envelope condition
    return Va_new, a1, c

def steady_state_policy(Pi,a_space,w,b,r,tau,eta,beta,tol):
    #initial guess of Va_init: assuming consume 5% of cash on hand
    coh_guess = income(w,b,tau) + (1+(1-tau)*r) * a_space
    c_guess = 0.05 * coh_guess
    Va_guess = (1+(1-tau)*r) * c_guess**(-eta)
    Va = Va_guess
    #Iteration until error of new and old asset below tolerance:
    for _ in range(10000): #Maximum steps of iteration: 10000
        Va, a, c = backward_iteration(Va,Pi,w,b,tau,r,beta,eta)
        if _>0 and np.max(np.abs(a - a_old)) < tol:
            print('Backward iteration stops at step '+str(_)+'. Error: '+str(np.max(np.abs(a - a_old))))
            return Va, a, c
        else:
            a_old = a
            continue

#Example:
w=1
b=1.199
tau=0
#income(w,b,tau)
#array([[0.5],
#       [1. ]])
eta=2
beta=0.995
r=0.02  
tol=1E-9
alpha=0.36
delta=0.005
#Va_ss, a_ss ,c_ss = steady_state_policy(Pi,a_space,w,b,r,tau,eta,beta,tol)


#Obtaining invariant/ steady state distribution across states (e,a)
#Size of states: (e,a) (2,100), 200 individual states
#CDF of distribution f(e,a)


#Lottery method
@numba.njit
def forward_iteration(a_ss, a_space, f0):
    f1 = np.zeros_like(f0)
    for e in range(2):
        for i in range(len(a_space)):
            #Search for the interval of household [e,i]'s asset in the asseet grid space
            for n in range(len(a_space)):
                if a_ss[e,i] >= a_space[n] and a_ss[e,i] <= a_space[n+1]:
                    #print(e,i,a_ss[e,i],n)
                    interval = [n,n+1]
                    break
                else:
                    continue
            
            weight = (a_space[interval[1]] - a_ss[e,i])/(a_space[interval[1]] - a_space[interval[0]])
            f1[e,interval[0]] += weight * f0[e,i]
            f1[e,interval[1]] += (1-weight) * f0[e,i]
    return Pi.T @ f1

def steady_distribution(a_ss, a_space, tol):
    #initiate uniform distribution
    f0 = pi.reshape((2, 1)) * np.ones_like(a_space) / len(a_space)
    #initiate forward distribution
    for _ in range(1000000000):
        f1 = forward_iteration(a_ss, a_space, f0)
        if np.max(np.abs(f1-f0)) < tol:
            print('Forward iteration stops at step '+str(_)+'. Error: '+str(np.max(np.abs(f1-f0))))
            return f1
            break
        f0 = f1

#example:
#f = steady_distribution(a_ss, a_space, tol)


#Steady state of individual and aggregate variables
#Step 1 - 8

#Step 1: stationary labor N
N=0.5/(1.5-0.9565)

#Step 2: initial guess of K and tau
K_init = 200
tau_init = 0.02

#Step 3: wage rate and rent
#Production:
def Yfn(N,K,alpha):
    return N**(1-alpha) * K**alpha

def rfn(N,K,alpha,delta):
    return alpha * (N/K)**(1-alpha) - delta

def wfn(N,K,alpha):
    return (1-alpha) * (K/N)**alpha

def Bfn(w,N,r,K,tau):
    return (1-N) * b

def Cfn(f, c_ss):
    C = 0
    for e in range(2):
        for a in range(len(a_space)):
            C += f[e,a] * c_ss[e,a]
    return C

#Step 4: Household decisions
w=wfn(N,K_init,alpha)
r=rfn(N,K_init,alpha,delta)
B=Bfn(w,N,r,K_init,tau_init)
Va_ss, a_ss ,c_ss = steady_state_policy(Pi,a_space,w,b,r,tau,eta,beta,tol)

#Step 5: Distribution on household states
f = steady_distribution(a_ss, a_space, tol)

#Step 6: Compute K and taxes T that solve the aggregate consistency conditions
def Kfn(f, a_ss):
    K = 0
    for e in range(2):
        for a in range(len(a_space)):
            K += f[e,a] * a_ss[e,a]
    return K

def Tfn(w,N,r,K,tau):
    return tau * (w*N + r*K)

K=Kfn(f, a_ss)

#Step 7: Compute the tax rate tau that solves gov budget balance
#tau = optimize.brentq(lambda tau: Tfn(w,N,r,K,tau) - B, -0.5, 0.5)
tau=0.0

#Step 8: Update K and tau
klist=[]
slist=[]

for _ in range(10000):
    w=wfn(N,K_init,alpha)
    r=rfn(N,K_init,alpha,delta)
    B=Bfn(w,N,r,K_init,tau)
    Va_ss, a_ss ,c_ss = steady_state_policy(Pi,a_space,w,b,r,tau,eta,beta,tol)
    f = steady_distribution(a_ss, a_space, tol)
    K=Kfn(f, a_ss)
    C=Cfn(f, c_ss)
    Y=Yfn(N,K,alpha)
    T=Tfn(w,N,r,K,tau)
    if abs(K-K_init) < 1E-7: #Y - C - K * delta
        print('Market clears at step '+str(_)+'. Steady State K :')
        print(K)
        break
    klist.append(K)
    slist.append(_)
    v = 0.005
    #K_init = (K+K_init)/2
    K_init = v * K + (1-v) * K_init
    #tau = optimize.brentq(lambda tau: Tfn(w,N,r,K,tau) - B, 0, 0.5)
plt.plot(slist,klist)
plt.show()

def plot():
    plt.plot(a_space,f[1,:])
    plt.plot(a_space,f[0,:])
    plt.show()

#Summary of steady state
ss={'Y':Y,
    'K':K,
    'N':N,
    'w':w,
    'r':r,
    'T':T,
    'B':B,
    'c':c_ss,
    'a':a_ss,  
    }
