import numpy as np
import numpy.random as rnd

# -----------------------------------------------------------------------------------------------------------------------
# Energy functions
# -----------------------------------------------------------------------------------------------------------------------

def spin_site_energy(self, spin_site_x, spin_site_y, grid_spins):
    '''Gives the energy of one single spin site
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    spin_site_x : float
        x coordinate of spin site
    spin_site_y : float
        y coordinate of spin site
    grid_spins : 2D array (L, L)
        containing al the spin values within the grid
           
    Returns
    -------
    spin_site_energy: float
        energy of a single spin site
        
    '''
    spin_site_energy = 0
    
    spin_neigbour_x = (spin_site_x + np.array([1, 0, -1, 0]))%(self.L)
    spin_neigbour_y = (spin_site_y + np.array([0, -1, 0, 1]))%(self.L)
    
    for i in range(np.size(spin_neigbour_x)):
        spin_value_center = grid_spins[spin_site_x, spin_site_y]
        spin_value_neighbour = grid_spins[spin_neigbour_x[i], spin_neigbour_y[i]]

        spin_site_energy += -self.J*spin_value_center*spin_value_neighbour - self.h*spin_value_center
        
    return spin_site_energy 


def system_energy(self, grid_coordinates, grid_spins, spin_site_numbers):
    '''Gives the energy of the system
    
    Parameters
    ----------
    grid_coordinates : 2D array (dim, L*L)
        containing the x coordinates of the spins        
    grid_spins : 2D array (L, L)
        containing al the spin values within the grid        
    spin_site_numbers: range (spin_site_total_number)
        containing integer counters up to size of spin_site_total_number
        
    Returns
    -------
    sys_energy: float
        the systems energy
        
    '''
    
    sys_energy = 0
    for spin_site_number in spin_site_numbers:
        spin_site_x = grid_coordinates[0][spin_site_number]
        spin_site_y = grid_coordinates[1][spin_site_number]
        sys_energy += spin_site_energy(self, spin_site_x, spin_site_y, grid_spins)
    sys_energy = sys_energy/2 # To counter double counting of the links
    
    return sys_energy