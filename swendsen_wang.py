import numpy as np
import numpy.random as rnd

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