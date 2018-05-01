import numpy as np
import numpy.random as rnd

# -----------------------------------------------------------------------------------------------------------------------
# System quantities functions
# -----------------------------------------------------------------------------------------------------------------------

def btstrp_rnd_gen(self):
    '''Generates sequence of random data points that can be used in the bootstrap algorithm.'''
    N = self.eq_data_points - 1
    a = np.round(np.random.random(self.bs_trials*N).reshape(self.bs_trials,N)*N)
    return a.astype(int)
    
def m_calculate(self, grid_spins):
    '''Gives magnetisation of the system
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    grid_spins : 2D array (L, L)
        containing al the spin values within the grid
    
    Returns
    -------
    magnetisation : float
        contains magnetisation of the system
        
    '''
    magnetisation = np.mean(grid_spins)
    
    return magnetisation

def chi_calculate(self, grid_spins):
    '''Gives susceptibility of the system
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    magnetisation_i : 1D array (1, eq_data_points)
        containing the past magnetisations for 
        the required ammount of equilibrium data
    
    Returns
    -------
    chi : float
        contains susceptibility of the system
        
    '''
    return (np.sum(grid_spins)**2)/self.L**4
    #return (self.spin_site_total_number)*np.var(magnetisation_i[-self.eq_data_points:])/((self.T)*(self.kb))

def c_v_calculate(self, energy_i, btstrp_seq):
    '''Gives specific heat of the system

    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    energy_i : 1D array (1, eq_data_points)
        containing the past energies for 
        the required ammount of equilibrium data

    Returns
    -------
    c_v : float
        contains specific heat of the system

    '''
    data_eq = energy_i[-self.eq_data_points:]
    c_v_temp = np.zeros((self.bs_trials, 1), dtype=float)
    for j in range(self.bs_trials):        
        data_sample = data_eq[btstrp_seq[j]]
        c_v_temp[j] = np.var(data_sample)/((self.spin_site_total_number)*(self.T**2)*(self.kb))

    c_v_ave = np.mean(c_v_temp)
    c_v_sig = np.sqrt(np.mean(c_v_temp**2) - c_v_ave**2)

    return c_v_ave, c_v_sig