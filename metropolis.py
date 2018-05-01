import numpy as np
import numpy.random as rnd
from energies import spin_site_energy

# -----------------------------------------------------------------------------------------------------------------------
# Spinflip functions
# -----------------------------------------------------------------------------------------------------------------------
def spin_flip_accept(self, spin_flip_energy_delta):
    '''If this difference, spin_flip_energy_delta (dE), is < 0 this flip is accepted,
    if dE > 0 this flip is accepted with probability e^-dE/K_bT.
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    spin_flip_energy_delta : float
        containing the energy difference upon a spin flip
        
    Returns
    -------
    boolean
        True if flip is accepted, false if rejected
    '''    
     
    return rnd.binomial(1, np.exp(-spin_flip_energy_delta/(self.kb*self.T))) == 1
    
def spin_flip_random(self, grid_coordinates, grid_spins, E_system):
    '''Uses an algorithm where is spin is chosen at random, flipped, the energy 
    difference is checked. If this difference is < 0 this flip is accepted,
    if dE > 0 this flip is accepted with probability e^-dE/K_bT.
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    grid_coordinates : 2D array (dim, L*L)
        containing the x coordinates of the spins
    grid_spins : 2D array (L, L)
        containing al the spin values within the grid
    E_system: float
        current value of the system energy

    Returns
    -------
    grid_spins : 2D array (L, L)
        containing al the spin values 
    E_system: float 
        energy of the system, which may have changed upon spin flip
    '''  
    
    spin_site_number = rnd.randint(0, self.spin_site_total_number)
    spin_site_x = grid_coordinates[0][spin_site_number]
    spin_site_y = grid_coordinates[1][spin_site_number]
       
    spin_site_energy_pre_flip = spin_site_energy(self, spin_site_x, spin_site_y, grid_spins)
    grid_spins[spin_site_x, spin_site_y] = -1 * grid_spins[spin_site_x, spin_site_y]
    spin_site_energy_post_flip = spin_site_energy(self, spin_site_x, spin_site_y, grid_spins) 
    
    spin_flip_energy_delta = spin_site_energy_post_flip - spin_site_energy_pre_flip
        
    if spin_flip_energy_delta <= 0:
        E_system += spin_flip_energy_delta
        
    elif spin_flip_energy_delta > 0:
        
        if  spin_flip_accept(self, spin_flip_energy_delta):
            E_system += spin_flip_energy_delta
            
        else:  
            grid_spins[spin_site_x, spin_site_y] = -1 * grid_spins[spin_site_x, spin_site_y]
            
    return grid_spins, E_system

