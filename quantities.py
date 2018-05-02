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

def chi_calculate(self, magnetisation, btstrp_seq):
    '''Gives susceptibility of the system
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    magnetisation: 1D array (1, eq_data_points)
        containing the past magnetisations for 
        the required ammount of equilibrium data
    btstrp_seq: 2D array (trials, eq_data_points)
        contains random sequence of samples that can be taken from the data, 
        for the ammount of bootstrap trials that are required
        
    Returns
    -------
    chi_ave : float
        contains the average susceptibility of the system
    chi_sig : float
        contains the sigma of the specific susceptibility of the system  
        
    '''
    
    data_eq = magnetisation[-self.eq_data_points:]
    chi_temp = np.zeros((self.bs_trials, 1), dtype=float)
    for j in range(self.bs_trials):        
        data_sample = data_eq[btstrp_seq[j]]        
        chi_temp[j] = (self.spin_site_total_number)*np.var(data_sample)/((self.T)*(self.kb))

    chi_ave = np.mean(chi_temp)
    chi_sig = np.std(chi_temp)
    
    return chi_ave, chi_sig

def c_v_calculate(self, energy_i, btstrp_seq):
    '''Gives specific heat of the system

    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    energy_i : 1D array (1, eq_data_points)
        containing the past energies for 
        the required ammount of equilibrium data
    btstrp_seq: 2D array (trials, eq_data_points)
        contains random sequence of samples that can be taken from the data, 
        for the ammount of bootstrap trials that are required

    Returns
    -------
    c_v_ave : float
        contains the average specific heat of the system
    c_v_sig : float
        contains the sigma of the specific heat of the system    
    
    '''
    data_eq = energy_i[-self.eq_data_points:]
    c_v_temp = np.zeros((self.bs_trials, 1), dtype=float)
    for j in range(self.bs_trials):        
        data_sample = data_eq[btstrp_seq[j]]
        c_v_temp[j] = np.var(data_sample)/((self.spin_site_total_number)*(self.T**2)*(self.kb))

    c_v_ave = np.mean(c_v_temp)
    c_v_sig = np.std(c_v_temp)

    return c_v_ave, c_v_sig