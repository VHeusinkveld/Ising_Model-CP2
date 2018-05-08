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
    '''    
    Simulation function for the ising model. Metropolis of Swendsen-Wang 
    can be chosen. The mean quared magnetisation, magnetic susceptibility and 
    specific heat are calculated and their uncertainty determined using 
    bootstrapping. Simulation can be run for different temperatures
    with interval dT. 
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
        
    Returns
    -------
    results : NameSpace
        contains all the simulation results  
        
    '''
    
    # Check input
    input_check(self)
    
    # Initialization
    grid_spins = assign_spin(self)
    grid_coordinates, spin_site_numbers = grid_init(self)
    T_total, h_total, energy, chi, c_v, magnetisation, energy_i, magnetisation_i, chi_i = matrix_init(self)
    
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