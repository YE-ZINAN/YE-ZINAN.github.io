import numpy as np
import numba
from scipy import optimize
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math

# some useful plot defaults
plt.rcParams.update({'font.size' : 20, 'lines.linewidth' : 3.5, 'figure.figsize' : (13,7)})

#Krusell Smith 1998, but endogenous grids
#Calibration
#Productivity in good time and bad time
Zg = 1.01
Zb = 0.99

#Trandisition metrics of Z
PiZ = np.array([[0.8,0.2],
                [0.2,0.8]])
#Transition metrics of employment state, with both today and tomorrow in good time
PiZg = np.array([[0.9615,0.0385],
                [0.9581,0.0419]])
#Transition metrics of employment state, with both today and tomorrow in bad time
PiZb = np.array([[0.9525,0.0475],
                [0.3952,0.6048]])

#Definition of the large transition metrics with Z and employment
'''Pi =   ([[bebe,bebu,bege,begu],
            [bube,bubu,buge,bugu],
            [gebe,gebu,gege,gegu],
            [gube,gubu,guge,gugu]])'''

#unemployment rate in good time and bad time
ug = 0.03863135
ub = 0.10729614

#Large transition metrics
Pi = ([[0.9525*0.8,0.0475*0.8,1*0.2,0*0.2],
       [0.3952*0.8,0.6048*0.8,(1-ug/ub)*0.2,ug/ub*0.2],
       [(1-ub)/(1-ug)*0.2,(1-(1-ub)/(1-ug))*0.2,0.9615*0.8,0.0385*0.8],
       [0*0.2,1*0.2,0.9581*0.8,0.0419*0.8]])

Pi = np.array(Pi)

#Transition metrics of employment state, with today in bad time and tomorrow in good time
PiZbZg = np.array([[1,0],[(1-ug/ub),ug/ub]])
#Transition metrics of employment state, with today in good time and tomorrow in bad time
PiZgZb = np.array([[(1-ub)/(1-ug),(1-(1-ub)/(1-ug))],[0,1]])

amin = 0      #Lower bound of individual asset level
amax = 12     #upper bound of individual asset level
na = 101      #Grid point number of individual asset
kmin = 3      #Lower bound of aggregate asset level
kmax = 9      #Upper bound of aggregate asset level
nk = 10       #Grid point number of aggregate asset
alpha = 0.36
beta = 0.96
delta = 0.1
eta = 1.5
tau = 0
b = 0.0001    #Unemployment transfer
tol = 1E-7    #Tolerance of optimal policy 

#Grid space for individual and aggregate asset
a_space = np.linspace(amin,amax,na)
k_space = np.linspace(kmin,kmax,nk)

#2/3. Initial guess on law of motion of K and the parameters
def H(K,y0,y1):
    return np.exp(y0 + y1 * np.log(K))

yb0 = 0.2
yb1 = 0.75
yg0 = 0.2
yg1 = 0.8

#Model:
#1. One-to-one mapping from Z to N
#Labor input
def Nfn(Z): 
    if Z == Zg:
        return 1-ug
    else:
        return 1-ub
    
def rfn(Z,K,alpha,delta):
    N=Nfn(Z)
    return Z * alpha * (N/K)**(1-alpha) - delta

def wfn(Z,K,alpha):
    N=Nfn(Z)
    return Z * (1-alpha) * (K/N)**alpha

#Income function
def income(Z,b,tau,K,alpha):
    N = Nfn(Z)
    w = wfn(Z,K,alpha)
    income = np.array([[(1-tau)*w],[b],[(1-tau)*w],[b]])
    return income #[4,1] metrix, [be,bu,ge,gu]'

#4. Solve the value function and policy function with backward iteration
def backward_iteration(Va,Pi,b,tau,K,beta,eta,delta,yb0,yb1,yg0,yg1):
    #Va: Value function taking derivative of a (from last iteration)
    #Va1: Value function taking derivative of a (new), metrics: [nk,4,na]
    Va1 = np.empty_like(Va)

    #Loop all tfp and employment states:
    #e == 0: bad time & employed
    #e == 1: bad time & unemployed
    #e == 2: good time & employed
    #e == 3: good time & unemployed
    #yb0: constant parameter of law of motion during bad time
    #yb1: main parameter of law of motion during bad time
    #yg0: constant parameter of law of motion during good time
    #yg1: main parameter of law of motion during good time
    #k1: expected aggregate K next one period, output of law of motion
    #interp_K: function, given aggregate k, find kj<= k <=kj+1 output w, k
    #w: the weight assigned to grid point kj (1-w to kj+1)
    #k: the rank of kj (k+1 is the rank of kj+1)
    for e in range(4):
        #During bad time using the law of motion with bad time parameters
        if e==0 or e == 1:
            #
            for kt in range(len(k_space)):
                k1 = H(k_space[kt],yb0,yb1)
                we, k = interp_K(k1,yb0,yb1)
                Va1[kt,e,:] = we * Va[k,e,:] + (1-we)* Va[k+1,e,:]
        #During good time using the law of motion with bad time parameters
        else:
            for kt in range(len(k_space)):
                k1 = H(k_space[kt],yg0,yg1)
                we, k = interp_K(k1,yg0,yg1)
                Va1[kt,e,:] = we * Va[k,e,:] + (1-we)* Va[k+1,e,:]

    #Taking expectation of the next period tfp and employment states, outer loop of nk
    for kt in range(len(k_space)):
        Va1[kt] = Pi @ Va1[kt]
    
    c_endog = (beta * Va1)**(-1/eta) #consumption with endogenous grid points, [nk,4,na]
    coh = np.zeros_like(Va) #Cash on hand, the RHS of constrints, [nk,4,na], row 0/1 with Zb, 2/3 with Zg
    for k in range(len(k_space)):
        K = k_space[k]
        for Z in [Zb,Zg]:
            if Z == Zb:
                w = wfn(Z,K,alpha)
                r = rfn(Z,K,alpha,delta)
                y = income(Z,b,tau,K,alpha)
                coh[k,0,:] = y[0] + (1+(1-tau)*r) * a_space
                coh[k,1,:] = y[1] + (1+(1-tau)*r) * a_space
            else:
                w = wfn(Z,K,alpha)
                r = rfn(Z,K,alpha,delta)
                y = income(Z,b,tau,K,alpha)
                coh[k,2,:] = y[2] + (1+(1-tau)*r) * a_space
                coh[k,3,:] = y[3] + (1+(1-tau)*r) * a_space
                
    #Generating a' policy funtion by interpolating c_endog+a' with cash on hand
    a1 = np.empty_like(coh)
    for k in range(len(k_space)):
        for e in range(4):
            a1[k,e,:]=np.interp(coh[k,e,:], c_endog[k,e,:] + a_space, a_space) #linear interpolation
            
    a1 = np.maximum(a1, a_space[0]) #limitation on borrowing
    c = coh - a1 #Backout consumption policy function
    Va_new = np.zeros_like(Va) #Generating new Va with envelope condition
    for k in range(len(k_space)):
        K = k_space[k]
        for e in range(4):
            if e==0 or e==1: #Bad time
                r = rfn(Zb,K,alpha,delta)
                Va_new[k,e,:] = (1+(1-tau)*r) * c[k,e,:]**(-eta)
            else:            #Good time
                r = rfn(Zg,K,alpha,delta)
                Va_new[k,e,:] = (1+(1-tau)*r) * c[k,e,:]**(-eta)
    return Va_new, a1, c

#Interpolation of aggregate K
def interp_K(K,y0,y1):
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

#Solving optimal value funtion and policy function
def steady_state_policy(Pi,a_space,b,tau,eta,beta,tol,delta,k_space,Hs,yb0,yb1,yg0,yg1):
    #initial guess of Va_init: assuming consume 5% of cash on hand
    Va = np.zeros((nk,4,na)) #[K,Pi,a] = (10,4,101)
    a = np.zeros((nk,4,na))
    c = np.zeros((nk,4,na))
    a_old = np.zeros((nk,4,na))
    for k in range(len(k_space)):
        w = wfn(1-ug,k_space[k],alpha)
        r = rfn(1-ug,k_space[k],alpha,delta)
        coh_guess = income(Zg,b,tau,k_space[k],alpha).reshape((4,1)) + (1+(1-tau)*r) * a_space
        c_guess = 0.05 * coh_guess
        Va_guess = (1+(1-tau)*r) * c_guess**(-eta)
        Va[k] = Va_guess
    k1 = k_space[2]
    #Iteration until error of new and old asset below tolerance, otherwise continue backward iteration
    for _ in range(10000): #Maximum steps of iteration: 10000
        Va, a, c = backward_iteration(Va,Pi,b,tau,k1,beta,eta,delta,yb0,yb1,yg0,yg1)
        if _>0 and np.max(np.abs(a - a_old)) < tol:
            if Hs==0:
                print('Backward iteration stops at step '+str(_)+'. Error: '+str(np.max(np.abs(a - a_old))))
            break
        else:
            a_old = a
            continue
    return Va, a, c

Hs=0
Va ,a, c = steady_state_policy(Pi,a_space,b,tau,eta,beta,tol,delta,k_space,Hs,yb0,yb1,yg0,yg1)     

#5. Simulation of distributio dynamics
@numba.njit
def forward_iteration(a,wk, kt, a_space, f0,Z,Pi,Z0):
    #Lottery method forward iteration

    #wk: the weight assigned to grid point kj (1-w to kj+1)
    #kt: the rank of kj (k+1: the rank of kj+1)
    #f0: distribution of last period
    #f1: today's distribution. [2,na]: [employment state, na]
    #Z: today's realized produtivity level
    #Z0: yestoday's productivity level
    #a: optimal policy function for a'(k,Z,e,a)
    f1 = np.zeros_like(f0)

    if Z == Zb:
        a = a[:,0:2,:] #(be, bu)
        for e in range(2):
            for k in [kt,kt+1]:
                for i in range(len(a_space)):
                    a[k,e,i] = np.maximum(a[k,e,i],a_space[0])
                    a[k,e,i] = np.minimum(a[k,e,i],a_space[-1])
                    #Search the interpolation of individual asset
                    #an<= a[k,e,i] <=an+1
                    #wa: weight assigned to an

                    #Loop all grid points of individual asset
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
    else:
        a = a[:,2:4,:] #(ge, gu)
        for e in range(2):
            for k in [kt,kt+1]:
                for i in range(len(a_space)):
                    a[k,e,i] = np.maximum(a[k,e,i],a_space[0])
                    a[k,e,i] = np.minimum(a[k,e,i],a_space[-1])
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

    #Employment state transition: from last period Z0 to today's Z
    if Z0 == Zb and Z == Zb:
        f1 = PiZb.T @ f1
    elif Z0 == Zb and Z == Zg:
        f1 = PiZbZg.T @ f1
    elif Z0 == Zg and Z == Zg:
        f1 = PiZg.T @ f1
    elif Z0 == Zg and Z == Zb:
        f1 = PiZgZb.T @ f1
    return f1


T = 2000 #Simulation period
#Initial distributio at period 0
f_init = np.zeros((2,na)) #(2,101): [employment_states, asset]
for e in range(2):
    for n in range(na):
        if n<na/2:
            f_init[e,n] = 1/2/na/2
        else:
            f_init[e,n] = 0
flist=[f_init]

#Initial mean K
K = 0
for e in range(2):
    K += np.vdot(f_init[e,:],a_space)

#Simulated path of Z
def generate_Z_path(PiZ, Zg,Zb, T):
    #Setting initial period to good time
    Z_path = [Zg]
    current_state = Zg
    for _ in range(T-1):
        if current_state == Zg:
            next_state = np.random.choice([Zg, Zb], p=[0.8,0.2])
            Z_path.append(next_state)
            current_state = next_state
        else:
            next_state = np.random.choice([Zb, Zg], p=[0.8,0.2])
            Z_path.append(next_state)
            current_state = next_state
    return Z_path

def plot_kpath(T):
    plt.plot(np.linspace(500,T,T-500),K_path[500:],label='K',linewidth = 0.8)
    plt.xlabel('Simulation periods')
    plt.legend()
    plt.show()

#plot_kpath(T)

#6. Estimation of law of motion H with OLS
def ols(y,x):
    x = sm.add_constant(x)
    model = sm.OLS(y,x)
    results = model.fit()
    r2 = results.rsquared
    coef = results.params
    return r2, coef


#7/8. Interate until parameters of H converge
def H_converge():

    T=3000    #Total periods
    Hs=1      #=1 doesn't print backward iteration results
    yb0 = 1.4 #Initial guess of law of motion parameters
    yb1 = 0
    yg0 = 1.5
    yg1 = 0

    #Setting first 10 period to good for illustration
    Z_path = generate_Z_path(PiZ, Zg, Zb, T)
    Z_path[:10] = [Zg for i in range(10)]

    #Separating good and bad periods
    Zg_list, Zb_list = [], []
    for z in range(500,len(Z_path)-1):
        if Z_path[z] == Zg:
            Zg_list.append(z)
        else:
            Zb_list.append(z)

    #Initial distribution, make sure the sum is 1
    f_init = np.zeros((2,na))
    for e in range(2):
        for n in range(na):
            if n<na/2:
                f_init[e,n] = 1/2/int(na/2)
            else:
                f_init[e,n] = 0

    #Initialize aggregate K with initial distribution
    K = 0
    for e in range(2):
        K += np.vdot(f_init[e,:],a_space)

    #Loop till law of motion parameters converge
    #Maximum loop set 10000, aviod non-stopping
    for i in range(10000):
        Va ,a, c = steady_state_policy(Pi,a_space,b,tau,eta,beta,tol,delta,k_space,Hs,yb0,yb1,yg0,yg1)     
        flist=[f_init]
        #Initial mean K = 6
        K_path =[K]

        #Simulate the path of distribution and aggregate K (first moment)
        for _ in range(1,T):
            Z = Z_path[_]
            Z0 = Z_path[_-1]
            K = K_path[-1]
            if Z == Zg:
                wt, kt = interp_K(K,yg0,yg1)
            if Z == Zb:
                wt, kt = interp_K(K,yb0,yb1)
            #f = forward_iteration(a[:,0:2,:], wt, kt, a_space, flist[_-1],PiZb)
            f = forward_iteration(a,wt, kt, a_space, flist[-1],Z,Pi,Z0)
            flist.append(f)
            K = 0
            for e in range(2):
                K += np.vdot(f[e,:],a_space)
            K_path.append(K)

        #Estimate new parameters for good time and bad time, respectively
        r2g, coefg = ols(np.log([K_path[z+1] for z in Zg_list]),np.log([K_path[z] for z in Zg_list]))
        r2b, coefb = ols(np.log([K_path[z+1] for z in Zb_list]),np.log([K_path[z] for z in Zb_list]))
        
        #Check if parameters converged
        new_paras = np.array([coefg[0],coefg[1],coefb[0],coefb[1]])
        old_paras = np.array([yg0,yg1,yb0,yb1])
        if np.max(np.abs(new_paras - old_paras)) < 1E-5:
            print('\n')
            print('H parameters converge at interation '+str(i+1))
            print('R2 of good time: '+str(r2g)+'. Estimated law of motion: ln(K\') = '+str(coefg[0])+' + '+str(coefg[1])+'ln(K)')
            print('R2 of bad time: '+str(r2b)+'. Estimated law of motion: ln(K\') = '+str(coefb[0])+' + '+str(coefb[1])+'ln(K)')
            return K_path, Zg_list, Zb_list, flist, Va, a ,c
            break
        else:
            print('Interation step: '+str(i+1)+'. Parameters of good time and bad time:')
            print(new_paras)
            print('R2 of good time and bad time:')
            print(r2g,r2b)
            print('Maximum error: '+str(np.max(np.abs(new_paras - old_paras)))+'\n')

            #Updating new parameters
            #Changing speed of updating, faster at first, slower afterward
            if i <7:
                v = 0.7
            else:
                v = 0.2
            yg0 = coefg[0] * v + yg0 * (1-v)
            yg1 = coefg[1] * v + yg1 * (1-v)
            yb0 = coefb[0] * v + yb0 * (1-v)
            yb1 = coefb[1] * v + yb1 * (1-v)

#Plot accordingly
K_path, Zg_list, Zb_list, flist, Va, a ,c = H_converge()


















