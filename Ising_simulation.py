import numpy as np
import numpy.random as rnd

from types import SimpleNamespace
from importlib import reload

from initialisation import *
from energies import *
from swendsen_wang import *
from metropolis import *
from quantities import *

# -----------------------------------------------------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------------------------------------------------
def IM_sim(self):
    
    # Check input
    input_check(self)
    
    # Initialization
    grid_spins = assign_spin(self)
    grid_coordinates, spin_site_numbers = grid_init(self)
    T_total, h_total, energy, chi, c_v, magnetisation, int_cor_time, energy_i, magnetisation_i, chi_i = matrix_init(self)
    
    # Simulation 
    for j, temp in enumerate(range(self.T_steps)):
        # Loop over different temperatures with interval dT
        energy_ii = system_energy(self, grid_coordinates, grid_spins, spin_site_numbers)
                
        if self.algorithm == 'SF':
            # Regular metropolis algorithm, here called single spin flip.
            for i, t in enumerate(range(self.time_steps)):
                
                # Energy and magnetisation are stored every montecarlo time step
                if t%self.MCS == 0:
                    energy_i[int(i/self.MCS)] = energy_ii 
                    magnetisation_i[int(i/self.MCS)] = np.mean(grid_spins)
                
                grid_spins, energy_ii = spin_flip_random(self, grid_coordinates, grid_spins, energy_ii)
                
        elif self.algorithm == 'SW':
            # Swendsen Wang algorithm.
            for i, t in enumerate(range(self.MC_steps)):
                islands, grid_spins, cluster_flips = SW_algorithm(self, grid_coordinates, spin_site_numbers, grid_spins)
                
                energy_i[i] = system_energy(self, grid_coordinates, grid_spins, spin_site_numbers)
                magnetisation_i[i] = np.mean(grid_spins)
                     
        # Store data for specific T, h
        energy[j] = np.mean(energy_i[-self.eq_data_points:])
        m_squared = (magnetisation_i)**2

        btstrp_seq = btstrp_rnd_gen(self)
        magnetisation[j] = m_calculate(self, m_squared, btstrp_seq)
        chi[j] = chi_calculate(self, abs(magnetisation_i), btstrp_seq)
        c_v[j] = c_v_calculate(self, energy_i, btstrp_seq)
        #int_cor_time[j] = integrated_cor_time(self, energy_i, btstrp_seq)
        
        T_total[j] = self.T 
        h_total[j] = self.h
       
        # Increment T and h
        self.T = self.T + self.dT
        self.h = self.h + self.dh
        
    # Store simulation restuls 
    results = SimpleNamespace(temperature = T_total,
                              magnetic_field = h_total,
                              chi = chi,
                              energy = energy,
                              magnetisation = magnetisation,
                              c_v = c_v,
                              int_cor_time = int_cor_time,
                              grid_spins = grid_spins,
                              grid_coordinates = grid_coordinates
                             )
    if self.algorithm == 'SW':
        results.islands = islands
    
    return results


# -----------------------------------------------------------------------------------------------------------------------
# Functions under development
# -----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def integrated_cor_time(self, data, btstrp_seq):
        
    data_eq = np.reshape(data[-self.eq_data_points:],(-1, ))
    int_cor_time_temp = np.zeros((self.bs_trials, 1), dtype=float)
        
    for j in range(self.bs_trials):  
        data_sample_1 = data_eq[btstrp_seq[j,int(np.floor(self.eq_data_points/4)):int(np.floor(self.eq_data_points*3/4))]] 
        data_sample_1 -= np.mean(data_sample_1)
        
        data_sample_2 = data_eq[btstrp_seq[j]] 
        data_sample_2 -= np.mean(data_sample_2)
        
        corr = np.correlate(data_sample_1, data_sample_2, 'valid')
        corr = corr/max(corr)
        
        int_cor_time_temp[j] = sum(1/2*corr)
    
    int_cor_time_ave = np.mean(int_cor_time_temp)
    int_cor_time_sig = np.std(int_cor_time_temp)
 
    return int_cor_time_ave, int_cor_time_sig



# Autocorrelation function (Normalised)
def Autocorrelation(X):
    '''Calculates the integrated autocorrelation time approximation for a certain window
    
    Parameters
    ----------
    X : 2D array (N,1)
        Succesive estimates X(i) of a physical quantity X from MC-process
        
    Returns
    -------
    tau : float
        Integrated autocorrelation time approx for X
        
    '''
    
    steps = X.shape[0]
    
    # Initialise autocorrelation array
    C_AA = np.zeros((steps,), dtype=float)
    
    for t in range(steps):
        print(t)
        C_AA[t] = np.mean(X[0:steps - t]*X[t:steps]) - np.mean(X)**2 # From section 7.4 Jos' book and paper by Wolff (1998) (In Wolff it's numerator.)
    
    rho = C_AA/(np.mean(X**2)-np.mean(X)**2)
    R_window = rho[steps-1]*rho[steps-1 - 1]/(rho[steps-1 - 1] - rho[steps-1])
    rho_sum_window = np.sum(rho, axis=0)
    
    tau = 1/2 + rho_sum_window + R_window
    
    return tau
