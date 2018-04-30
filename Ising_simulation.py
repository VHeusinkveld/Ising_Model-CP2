import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from types import SimpleNamespace
from sys import exit

# -----------------------------------------------------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------------------------------------------------
def IM_sim(self):
    
    # Check input
    input_check(self)
    
    # Initialization
    grid_spins = assign_spin(self)
    grid_coordinates, spin_site_numbers = grid_init(self)
    magnetisation, T_total, h_total, energy, chi, c_v, energy_i, magnetisation_i = matrix_init(self)
    
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
                    magnetisation_i[int(i/self.MCS)] = magnetisation_f(self, grid_spins)
                    
                grid_spins, energy_ii = spin_flip_random(self, grid_coordinates, grid_spins, energy_ii)
                
        elif self.algorithm == 'SW':
            # Swendsen Wang algorithm.
            for i, t in enumerate(range(self.MC_steps)):
                islands, grid_spins, cluster_flips = SW_algorithm(self, grid_coordinates, spin_site_numbers, grid_spins)
                
                energy_i[i] = system_energy(self, grid_coordinates, grid_spins, spin_site_numbers)
                magnetisation_i[i] = magnetisation_f(self, grid_spins)
        
        # Store data for specific T, h
        energy[j] = np.mean(energy_i[-self.eq_data_points:])
        magnetisation[j] = np.mean(abs(magnetisation_i[-self.eq_data_points:]))
        chi[j] = chi_calculate(self, magnetisation_i)
        c_v[j] = c_v_calculate(self, energy_i)
        
        def correlation_func(self, X):
            if self.cor_cal:
                cor_data = np.reshape(X[-self.eq_data_points:] - np.mean(X[-self.eq_data_points:]), (-1,))   
                correlation = np.correlate(cor_data, cor_data, "full")
            else:
                correlation = 0
                
            return correlation  
        
        T_total[j] = self.T 
        h_total[j] = self.h
        
        # Autocorrelation

        # Increment T and h
        self.T = self.T + self.dT
        self.h = self.h + self.dh
    
    cor_energy = correlation_func(self, energy_i)
    cor_magnetisation = correlation_func(self, magnetisation_i)    
    
    # Store simulation restuls 
    results = SimpleNamespace(temperature = T_total,
                              magnetic_field = h_total,
                              chi = chi,
                              energy = energy,
                              magnetisation = magnetisation,
                              c_v = c_v,
                              cor_energy = cor_energy,
                              cor_magnetisation = cor_magnetisation
                             )
    
    return results, grid_coordinates

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
    
    magnetisation = np.zeros([self.T_steps, 1])
    T_total = np.zeros([self.T_steps, 1])
    h_total = np.zeros([self.T_steps, 1])
    energy = np.zeros([self.T_steps, 1])
    chi = np.zeros([self.T_steps, 1])
    c_v = np.zeros([self.T_steps, 1])
    
    energy_i = np.zeros([self.MC_steps, 1])
    magnetisation_i = np.zeros([self.MC_steps, 1])
       
    return magnetisation, T_total, h_total, energy, chi, c_v, energy_i, magnetisation_i

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


# -----------------------------------------------------------------------------------------------------------------------
# System parameter functions
# -----------------------------------------------------------------------------------------------------------------------
def magnetisation_f(self, grid_spins):
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

def chi_calculate(self, magnetisation_i):
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
    
    return (self.spin_site_total_number)*np.var(magnetisation_i[-self.eq_data_points:])/((self.T)*(self.kb))
             
def c_v_calculate(self, energy_i):
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
    
    return np.var(energy_i[-self.eq_data_points:])/((self.spin_site_total_number)*(self.T**2)*(self.kb))


# -----------------------------------------------------------------------------------------------------------------------
# Swendsen-Wang algorithm functions
# -----------------------------------------------------------------------------------------------------------------------
def SW_algorithm(self, grid_coordinates, spin_site_numbers, grid_spins):
    '''Functions that identifies al the clusters, "islands", 
    using the back_track function. At every iteration it is determined
    if the cluster is flipped with a 50/50 chance. The function 
    runs over all spins. Everytime a new cluster is identified it 
    gets added to the total cluster list.
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    grid_coordinates : 2D array (dim, L*L)
        containing the x coordinates of the spins
    bonds : 3D array (2, L, L)
        contains al the bonds present in the system.
    grid_spins : 2D array (L, L)
        containing al the spin values within the grid

    Returns
    -------
    islands : list
        contains list of the clusters which form islands of the same spin
    grid_spins : 2D array (L, L)
        containing al the spin values within the grid
    cluster_flips: list of length np.size(cluster)
        list which states if the cluster is flipped
    '''
    
    islands = []
    cluster_flips = []
    not_visited = np.ones((self.L, self.L), dtype= bool)
    
    bonds = bond_eval(self, grid_spins)
    
    for i in spin_site_numbers:
        cluster = []
        flip_cluster = 2*rnd.randint(2) - 1 
        spin_site_x = grid_coordinates[0][i]
        spin_site_y = grid_coordinates[1][i]
        cluster, grid_spins = back_track(self, spin_site_x, spin_site_y, bonds, not_visited, cluster, grid_spins, flip_cluster)
        if cluster != []:
            islands.append(cluster)
            cluster_flips.append(flip_cluster)
            
    return islands, grid_spins, cluster_flips


def bond_eval(self, grid_spins):
    '''Goes over all the spins in the system and checks the bonds, 
    if they are opposite the bond is set to 0 if they are equal the 
    bond is set to infinity with probability (1 - e^-2J/(k_bT)).
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    grid_spins : 2D array (L, L)
        containing al the spin values within the grid  


    Returns
    -------
    bonds : 3D array (2, L, L)
        contains al the bonds present in the system. The first 2D array
        gives the horizontal bonds (Element (i,j) gives the relation
        between spin_site (i,j) and (i,j+1). When j+1 does not exist it
        refers to (i,0) which illustrates the periodic BCs.) and the
        second 2D array gives the vertical bonds (Element  (i,j) gives
        the relation between spin_site (i,j) and (i+1,j). When i+1 does
        not exist it refers to (0,j) which illustrates the periodic BCs.).
    '''
    
    bonds = np.zeros((2, self.L, self.L,),dtype=float)
    chance_value = np.minimum(1, np.exp(-2*self.J/(self.kb*self.T)))
    
    delta_spin_hor = np.abs(grid_spins+np.roll(grid_spins,-1,axis=1))/2 # Divided by 2 to normalise
    nz_delta_spin_hor = np.asarray(np.nonzero(delta_spin_hor)) # Gives array with indices for non-zero elements

    delta_spin_ver = np.abs(grid_spins+np.roll(grid_spins,-1,axis=0))/2 # Divided by 2 to normalise
    nz_delta_spin_ver = np.asarray(np.nonzero(delta_spin_ver)) # Gives array with indices for non-zero elements

    for i in range(np.shape(nz_delta_spin_hor)[1]):
        if rnd.binomial(1, chance_value) == 1:
            bonds[0, nz_delta_spin_hor[0,i], nz_delta_spin_hor[1,i]] = 0
        else:
            bonds[0, nz_delta_spin_hor[0,i], nz_delta_spin_hor[1,i]] = np.inf

    for j in range(np.shape(nz_delta_spin_ver)[1]):
        if rnd.binomial(1, chance_value) == 1:
            bonds[1, nz_delta_spin_ver[0,j], nz_delta_spin_ver[1,j]] = 0
        else:
            bonds[1, nz_delta_spin_ver[0,j], nz_delta_spin_ver[1,j]] = np.inf
    
    return bonds


def back_track(self, x, y, bonds, not_visited, cluster, grid_spins, flip_cluster):
    '''Checks the neighbours of the spin, if they are 
    equal this functions jumps over to that spin and 
    repeats itself. The spins that are already visited 
    are skipped. Everytime an equal bond is found, this
    spind is added to the cluster.
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    x : float
        x coordinate of spin site
    y : float
        y coordinate of spin site
    bonds : 3D array (2, L, L)
        contains al the bonds present in the system.
    not_visited : 2D array (L, L)
        contains boolean for every spin site in system. True is not visited, false is visited
    cluster : list
        contains list of the coordinates belonging to one cluster
    grid_spins : 2D array (L, L)
        containing al the spin values within the grid
    flip_cluster: int 
        Value is 1 or -1, where -1 means a spinflip.

    Returns
    -------
    cluster : list
        contains list of the coordinates belonging to one cluster
    grid_spins : 2D array (L, L)
        containing al the spin values within the grid
    '''
    
    if not_visited[x, y]:
        not_visited[x, y] = False
        cluster.append([x, y])
        grid_spins[x, y] = grid_spins[x, y] * flip_cluster
                
        if bonds[0][x][y] == np.inf:
            n_x = x
            n_y = (y + 1)%self.L
            cluster, grid_spins = back_track(self, n_x, n_y, bonds, not_visited, cluster, grid_spins, flip_cluster)
            
        if bonds[0][x][(y - 1)%self.L] == np.inf:
            n_x = x
            n_y = (y - 1)%self.L
            cluster, grid_spins = back_track(self, n_x, n_y, bonds, not_visited, cluster, grid_spins, flip_cluster)
            
        if bonds[1][x][y] == np.inf:
            n_x = (x + 1)%self.L
            n_y = y
            cluster, grid_spins = back_track(self, n_x, n_y, bonds, not_visited, cluster, grid_spins, flip_cluster)
            
        if bonds[1][(x - 1)%self.L][y] == np.inf:
            n_x = (x - 1)%self.L
            n_y = y
            cluster, grid_spins = back_track(self, n_x, n_y, bonds, not_visited, cluster, grid_spins, flip_cluster)
            
    return cluster, grid_spins

# -----------------------------------------------------------------------------------------------------------------------
# Plot functions
# -----------------------------------------------------------------------------------------------------------------------
def grid_plot(x, y, S):
    '''Gives plot of the spins in the grid
    
    Parameters
    ----------
    x : 2D array (L,L)
        x coordinates of spin grid
    y : 2D array (L,L)
        y coordinates of spin grid
    S : 2D array (L,L)
        spin values in grid
        
    Returns
    -------
    image : matplotlib object
        Visualisation of the spin in the grid
        
    '''
    
    image = plt.imshow(S, extent=(x.min(), x.max(), y.max(), y.min()), interpolation='nearest', cmap=cm.plasma)
    plt.clim(-1,1)
    plt.xlabel('y')
    plt.ylabel('x')
    
    return image


def visualize_islands(self, islands):
    m_size = 47000/(self.L*self.L)
    # Visualize spin islands
    x_coord = []
    y_coord = []

    for i in islands:
        x_temp = []
        y_temp = []
        for x, y in i:
            x_temp.append(x)
            y_temp.append(y)

        x_coord.append(x_temp)     
        y_coord.append(y_temp)  

    for i, x in enumerate(x_coord):
        y = y_coord[i]
        plt.scatter(y, x, s=m_size, marker="s")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-0.5, self.L - 0.5])
    plt.ylim([self.L - 0.5, -0.5])
    
# -----------------------------------------------------------------------------------------------------------------------
# Functions under development
# -----------------------------------------------------------------------------------------------------------------------
import numpy as np

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
