import numpy as np

def f_chi(tau, factor, a):
    return factor*tau**a

def f_cv(tau, factor):
    return factor*np.log(tau)

def f_magnetisation(tau, factor, a):
    return factor*tau**a #(No minus at tau, as there is already taken care of in fitting function)
