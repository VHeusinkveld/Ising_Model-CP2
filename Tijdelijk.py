def f_chi(tau, factor, a, const):
    return factor*abs(tau)**(-a) + const

def f_cv(tau, factor, a, const):
    return factor*abs(tau)**(-a) + const
# in which tau = T-T_c

def f_magnetisation(tau, factor, beta, const):
    return factor*(-1*tau)**(beta) + const