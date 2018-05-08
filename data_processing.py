import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------------------------------------------------------
# Fit functions critical exponents and performance
# -----------------------------------------------------------------------------------------------------------------------

def f_chi(tau, factor, a):
    '''Trial fitting function magnetic susceptibility critical exponent.  
    
    Parameters
    ----------
    tau : 1D array 
        x data as input for the function 
    factor : float
        function parameter 
    a : float
        function parameter 
        
    Returns
    -------
    : function value 
        
    ''' 
    return factor*tau**a

def f_cv(tau, factor):
    '''Trial fitting function specific heat critical exponent.  
    
    Parameters
    ----------
    tau : 1D array 
        x data as input for the function 
    factor : float
        function parameter 
        
    Returns
    -------
    : function value 
        
    ''' 
    return factor*np.log(tau)

def f_magnetisation(tau, factor, a):
    '''Trial fitting function magnetisation critical exponent.  
    
    Parameters
    ----------
    tau : 1D array 
        x data as input for the function 
    factor : float
        function parameter 
    a : float
        function parameter 
        
    Returns
    -------
    : function value 
        
    '''    
    return factor*tau**a #(No minus at tau, as there is already taken care of in fitting function)

def f_z(L, x, c):
    '''Trial fitting function critical dynamical exponent.  
    
    Parameters
    ----------
    L : 1D array 
        x data as input for the function 
    x : float
        function parameter 
    c : float
        function parameter 
        
    Returns
    -------
    : function value 
        
    '''    
    return L**(x) + c

def fit_funct_z(f_z, L, Y, err, bounds):
    '''Function used for fitting to simulation perforance.  
    
    Parameters
    ----------
    f_z : func 
        the function to which should be fitted 
    L : 1D array
        contains x coordinates of the data
    Y : 1D array
        contains y coordinates of the data
    err : 1D array
        contains y errors
    bounds: 1D array
        contains bounds on the fitting parameters
        
    Returns
    -------
    popt: 1D array (2,)
        contains fitted values for the parameters 
        in the fit functions
    fit_err: 1D array (2,)
        corresponding errors to the fit values
        
    '''
    
    popt, pcov = curve_fit(f_z, L, Y, sigma = err, bounds = bounds)
    fit_err = np.sqrt(np.diag(pcov))
    return popt, fit_err

def f_sim(x, a, b, c):
    '''Trial fitting function simulation behaviour.  
    
    Parameters
    ----------
    x : 1D array 
        x data as input for the function 
    a : float
        function parameter 
    b : float
        function parameter 
    c : float
        function parameter 
        
    Returns
    -------
    : function value 
        
    '''     
    
    return a*x**(b) + c

def fit_funct_sim(f_sim, X, Y):
    '''Function used for fitting to simulation perforance.  
    
    Parameters
    ----------
    f_sim : func 
        the function to which should be fitted 
    X : 1D array
        contains x coordinates of the data
    Y : 1D array
        contains y coordinates of the data
        
    Returns
    -------
    popt: 1D array (2,)
        contains fitted values for the parameters 
        in the fit functions
    fit_err: 1D array (2,)
        corresponding errors to the fit values
        
    '''
    
    popt, pcov = curve_fit(f_sim, X, Y)
    fit_err = np.sqrt(np.diag(pcov))
    return popt, fit_err


# -----------------------------------------------------------------------------------------------------------------------
# Plot functions
# -----------------------------------------------------------------------------------------------------------------------
def grid_plot(self, results, fig_dir, identifier, save):
    '''Gives plot of the spins in the grid
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    results : NameSpace
        contains all the simulation results
    fig_dir : str
        directory where figures should be stored
    identifier : str
        identifier of the data set, is used in filename 
    save : bool
        determines if files should be saved
    '''
    
    figure_directory = fig_dir
    
    x = results.grid_coordinates[0]
    y = results.grid_coordinates[1]
    S = results.grid_spins
        
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=18)
    
    image = plt.imshow(S, extent=(x.min(), x.max(), y.max(), y.min()), interpolation='nearest', cmap=cm.plasma)
    plt.clim(-1,1)
    plt.xlabel('y')
    plt.ylabel('x')
    if save:
        plt.savefig(figure_directory + identifier + '_' +'grid_spins.png')
    plt.close()
    

def visualize_islands(self, results, fig_dir, identifier, save):
    '''Gives plot of the clusters in the grid. 
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    results : NameSpace
        contains all the simulation results  
    fig_dir : str
        directory where figures should be stored
    identifier : str
        identifier of the data set, is used in filename 
    save : bool
        determines if files should be saved
        
    '''
    if self.algorithm == 'SW':

        figure_directory = fig_dir

        islands = results.islands

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('font', size=18)

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
        plt.xlabel('y')
        plt.ylabel('x')

        if save:
            plt.savefig(figure_directory + identifier + '_' + 'clusters.png')
        plt.close()
    
def plot_func(self, results, fig_dir, identifier, save):
    '''Plots all required quantities, m, c_v, chi.
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    results : NameSpace
        contains all the simulation results  
    fig_dir : str
        directory where figures should be stored
    identifier : str
        identifier of the data set, is used in filename 
    save : bool
        determines if files should be saved
    '''

    figure_directory = fig_dir
    fig_name = figure_directory + identifier + '_' + self.algorithm + str(self.eq_data_points) 
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=18)
    
    x = results.temperature
    y = results.c_v[:,0]
    y_err = results.c_v[:,1]
    
    plt.errorbar(x, y, yerr=y_err, fmt='x', markersize=6, capsize=4)
    plt.xlabel('$\mathrm{k_B T/J}$', fontsize=18)
    plt.ylabel('$\mathrm{C_v}$', fontsize=18)
    plt.tight_layout()
    if save:
        plt.savefig(fig_name + '_cv.png')
    plt.close()

    x = results.temperature
    y = results.chi[:,0]
    y_err = results.chi[:,1]
    
    plt.errorbar(x, y, yerr=y_err, fmt='x', markersize=6, capsize=4)
    plt.xlabel('$\mathrm{k_B T/J}$', fontsize=18)
    plt.ylabel('$\chi$', fontsize=18)
    plt.tight_layout()
    if save:
        plt.savefig(fig_name + '_chi.png')
    plt.close()
    
    x = results.temperature
    y = results.magnetisation[:,0]
    y_err = results.magnetisation[:,1]
    plt.plot(x, y, 'x', markersize=6)
    #plt.errorbar(x, y, yerr=y_err, fmt='x', markersize=6, capsize=4)
    plt.xlabel('$\mathrm{k_B T/J}$', fontsize=18)
    plt.ylabel('$ m^{2} $', fontsize=18)    
    plt.tight_layout()
    if save:
        plt.savefig(fig_name + '_m_sq.png')
    plt.close()
    
    
    if save:
        print('Figures are saved to: ' + figure_directory)

# -----------------------------------------------------------------------------------------------------------------------
# Save data
# -----------------------------------------------------------------------------------------------------------------------
        
def save_data(self, results, data_dir, identifier):
    '''Saves most imported data to a npz file. This file
    can contain multiple numpy arrays.
    
    Example code of how to read data from a npzfile
       {
           npzfile = np.load(<filename>)
           npzfile.files
           npzfile[<quantity>]
       }
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    results : NameSpace
        contains all the simulation results        

    '''
    
    data_name = data_dir + 'saved_data_' + identifier + '_' + self.algorithm + '_' + str(self.eq_data_points)
    np.savez(data_name, temperature = results.temperature, magnetic_field = results.magnetic_field, 
             c_v = results.c_v, chi = results.chi, magnetisation = results.magnetisation,
            sim_time = results.sim_time)
    print('Data is saved to: ' + data_dir)