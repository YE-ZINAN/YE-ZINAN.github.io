import re


statelist = ['KG','Ks'] #
matlab_code = ''' 
exp(Kp) = (Ap * alpha / R)^(1/(1-alpha)) * exp(KG(-1))^(phi/(1-alpha));
exp(Yp) = Ap^(1/(1-alpha)) * (alpha/R)^(alpha/(1-alpha)) * exp(KG(-1))^(phi/(1-alpha));
exp(Ys) = As * exp(KG(-1))^phi * exp(Ks(-1))^alpha;
exp(Y) = exp(Ys) + exp(Yp);
exp(C) + exp(KG) + exp(Ks) = tau*R*exp(Kp) + exp(Ys) + (1-delta)*(exp(KG(-1))+exp(Ks(-1)));
exp(parKp_KG) = (Ap * alpha / R)^(1/(1-alpha)) * (phi/(1-alpha)) * exp(KG(-1))^(phi/(1-alpha)-1);
exp(parYs_KG) = As * phi * exp(KG(-1))^(phi-1) * exp(Ks(-1))^alpha;
exp(parYs_Ks) = As * exp(KG(-1))^phi * alpha * exp(Ks(-1))^(alpha-1);
1/exp(C) = beta * exp(chi(1)) * exp(parKp_KG(1)) + beta / exp(C(1)) * tau * R * exp(parKp_KG(1)) + beta / exp(C(1)) * (exp(parYs_KG(1)) + 1 -delta);
1/exp(C) = beta / exp(C(1)) * (exp(parYs_Ks(1)) + 1 - delta);

chi = (1-rho_chi) * log(chi_ss) + rho_chi*chi(-1) + echi;

exp(KGKs_Y) = (exp(KG(-1)) + exp(Ks(-1))) / exp(Y);
exp(Kp_Y) = exp(Kp) / exp(Y);
'''
#

slist = []
varlist = []
for s in matlab_code.split('\n'):
    if s != '' and s !=' ':
        slist.append(s)

for s in slist:
    varlist+=re.findall(r'exp\(([^()]+)\)', s)
varlist = list(set(varlist))

for s in slist:
    s = re.sub(r'exp\((.*?)\)', r'\1', s)
    s = s.replace(';','')
    s = s.replace('^','**')
    for state in statelist:
        if state not in s:
            continue
        elif state not in s and state+'(-1)' in s:
            s = s.replace(state+'(-1)',state)
        elif state in s and state+'(-1)' not in s:
            s = s.replace(state,state+'(1)')
        else:
            s = s.replace(state,state+'???(1)')
            s = s.replace(state+'???(1)(-1)',state+'???')
    s = s.replace('(1)(1)','(1)')
    s = s.replace('???','')
    print(s)

print('''

''',varlist)




