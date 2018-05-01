import numpy as np
import numpy.random as rnd
from sys import exit


# -----------------------------------------------------------------------------------------------------------------------
# Input parameter checks
# -----------------------------------------------------------------------------------------------------------------------

def input_check(self):
    '''This function check the input values for certain errors, 
    and exits the program with a error message.
    
    Parameters
    ----------
    self : NameSpace
    contains all the simulation parameters
    '''

    if self.eq_data_points > self.MC_steps:
        exit("More equilibrium data points then data points were selected.")
       
    if self.T + self.dT * self.T_steps < 0:
        exit("Selected T, T_steps and dT such that the temperature stays positive.")
    
    if self.algorithm == 'SW' and (self.h != 0 or self.dh != 0):
        exit("For SW the magnetic field should be zero.")
        
    if self.algorithm == 'SW' and self.L > 40:
        exit("This cluster size will exceed recursion depth for the backtrack algorithm.")
        
    if self.algorithm != 'SW' and self.algorithm != 'SF':
        exit("This is no valid algorithm: please select SF or SW.")
    
    if self.cor_cal != True and self.cor_cal != False:
        exit("cor_calc should be set to True or False")

# -----------------------------------------------------------------------------------------------------------------------
# Initialisation functions
# -----------------------------------------------------------------------------------------------------------------------

def assign_spin(self):
    '''Initialises the spin states in the system
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
           
    Returns
    -------
    grid_spins : 2D array (L, L)
        containing al the spin values within the grid
        
    '''
    
    if self.spin_init == 'random':
        grid_spins = rnd.rand(self.L, self.L)
        grid_spins[grid_spins >= 0.5] = 1
        grid_spins[grid_spins <  0.5] = -1
        
    elif self.spin_init == 'up':
        grid_spins = np.ones([self.L,self.L], dtype= int)
        
    elif self.spin_init == 'down':
        grid_spins = -1*np.ones([self.L,self.L], dtype= int)
        
    return grid_spins


def grid_init(self):
    '''Initialises the grid
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
           
    Returns
    -------
    grid_coordinates : mesh
    spin_site_numbers : range (spin_site_total_number)
        containing integer counters up to size of spin_site_total_number
        
    '''
    
    grid_x, grid_y = [range(self.L), range(self.L)]
    grid_coordinates = np.meshgrid(grid_x, grid_y) 
    grid_coordinates = np.reshape(grid_coordinates,(2,-1))
    spin_site_numbers = range(self.spin_site_total_number)
    
    return grid_coordinates, spin_site_numbers


def matrix_init(self):
    '''Initialises several parameter arrays
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
           
    Returns
    -------
    magnetisation : 2D array (T_steps, 1)
        initialised magnetisation array
    T_total : 2D array (T_steps, 1)
        initialised temperature array
    h_total : 2D array (T_steps, 1)
        initialised magnetic field array
    energy : 2D array (T_steps, 1)
        initialised energy array
    chi : 2D array (T_steps, 1)
        initialised susceptibility array
    c_v : 2D array (T_steps, 1)
        initialised specific heat array
     
        
    '''
    
    magnetisation = np.zeros([self.T_steps, 2])
    T_total = np.zeros([self.T_steps, 1])
    h_total = np.zeros([self.T_steps, 1])
    energy = np.zeros([self.T_steps, 1])
    chi = np.zeros([self.T_steps, 2])
    c_v = np.zeros([self.T_steps, 2])
    
    
    energy_i = np.zeros([self.MC_steps, 1])
    magnetisation_i = np.zeros([self.MC_steps, 1])
    chi_i = np.zeros([self.MC_steps, 1])
       
    return magnetisation, T_total, h_total, energy, chi, c_v, energy_i, magnetisation_i, chi_i
