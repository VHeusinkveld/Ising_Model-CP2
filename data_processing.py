import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------------------------------------------------------
# Data loading functions
# -----------------------------------------------------------------------------------------------------------------------
def load_data(data_dir, quantity):
    '''Loads all data from data_dir and assigns data of specific quantity to variables
    
    Parameters
    ----------
    data_dir : datafile ~.npz
        contains the 'temperature' 'magnetic_field' 'c_v' 'chi' 'magnetisation' arrays
    quantity : str
        quantity of interest
    
    Returns
    -------
    xdata : 1D array
        temperatures from data_dir
    ydata : 1D array
        quantity from data_dir
    y_err : 1D array
        standard deviation of ydata entries        
    '''
    
    data = np.load(data_dir)
    xdata = data['temperature'].reshape(np.shape(data['temperature'])[0])
    ydata = data[quantity][:,0]
    y_err = data[quantity][:,1]
    
    return xdata, ydata, y_err

# -----------------------------------------------------------------------------------------------------------------------
# Fit functions critical exponents and performance
# -----------------------------------------------------------------------------------------------------------------------

def f_chi(tau, factor, a):
    return factor*tau**a

def f_cv(tau, factor):
    return factor*np.log(tau)

def f_magnetisation(tau, factor, a):
    return factor*tau**a #(No minus at tau, as there is already taken care of in fitting function)

def f_z(L, x, c):
    return L**(x) + c

def fit_funct_z(f_z, L, Y, err, bounds):
    popt, pcov = curve_fit(f_z, L, Y, sigma = err, bounds = bounds)
    fit_err = np.sqrt(np.diag(pcov))
    return popt, fit_err

def f_sim(x, a, b, c):
    return a*x**(b) + c

def fit_funct_sim(f_sim, X, Y):
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
    '''Gives plot of the clusters in the grid
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    results : NameSpace
        contains all the simulation results        
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
    '''Plots all required quantities
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    results : NameSpace
        contains all the simulation results        
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

def plot_function(data_dir, quantity, LOG):
    '''Gives plot of the loaded data in loglog or normal scale. X-axis is
    translated with critical temperature T_c. (Done to get idea for fitting.)
    
    Parameters
    ----------
    data_dir : datafile ~.npz
        contains the 'temperature' 'magnetic_field' 'c_v' 'chi' arrays
    quantity : Str
        string which specifies which physical quantity must be fitted
    LOG : boolean
        Turns loglog plot on or off
        
    Returns
    -------
    plot of data provided in loglog or normal scale
    '''
    
    xdata, ydata, y_err = load_data(data_dir, quantity)
    
    T_c = crit_temp(data_dir)
    
    #if quantity == 'magnetisation':
    
    if LOG:
        plt.loglog(xdata[(xdata-T_c)>0]-T_c, ydata[(xdata-T_c)>0], 'bx')
        plt.loglog(abs(xdata[(xdata-T_c)<0]-T_c), ydata[(xdata-T_c)<0], 'rx')
        plt.grid()
    else:
        plt.plot(xdata[(xdata-T_c)>0]-T_c, ydata[(xdata-T_c)>0], 'bx')
        plt.plot(xdata[(xdata-T_c)<0]-T_c, ydata[(xdata-T_c)<0], 'rx')
        
    plt.tight_layout()
    plt.show()
    
# -----------------------------------------------------------------------------------------------------------------------
# Fit function for critical exponents & T_c function needed for fitting
# -----------------------------------------------------------------------------------------------------------------------

def crit_temp(data_dir):
    '''Gives the transition/critical temperature from the measured specific
    heat in terms of k_B*T/J.
    
    Parameters
    ----------
    data_dir : datafile ~.npz
        contains the 'temperature' 'magnetic_field' 'c_v' 'chi' arrays
    
    Returns
    -------
    T_c: float
        the critical temperature    
    '''
    
    data = np.load(data_dir)
    T = data['temperature'].reshape(np.shape(data['temperature'])[0])
    c_v = data['c_v'][:,0]
    T_c = T[c_v==max(c_v)]
    
    return T_c

def fit_function(data_dir, quantity, fit_range, plotYN, LOG):
    '''Gives the best fit to quantity with non-linear least squares
    
    Parameters
    ----------
    data_dir : datafile ~.npz
        contains the 'temperature' 'magnetic_field' 'c_v' 'chi' arrays
    quantity : Str
        string which specifies which physical quantity must be fitted
    fit_range: sequence
        [LB, UB], the lower and upper boundary of the interval to be
        fitted w.r.t T_c
    plotYN : boolean
        turns plotting of fit and original function on or off
    LOG : boolean
        Turns loglog plot on or off
              
    Returns
    -------
    popt: 1D array
        Optimal values for parameters so sum of the squared residuals is minimized
    fit_err : 1D array
        Standard deviation of the optimal values for the parameters
    xdata_fit_plot : 1D array
        Temperature values for plot in het fit domain
    ydata_fit_plot : 1D array
        Quantitiy values for plot in fit domain determined by applying the fitted curve
    '''
    
    # Initialisation of plot design and parameters
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=18)
    
    # Import needed functions from data_processing.py
    from data_processing import crit_temp, load_data, plot_function
       
    # Loading data and general functions (and generate transition temperature (T_c))
    from scipy.optimize import curve_fit
    
    data = np.load(data_dir)
    xdata = data['temperature'].reshape(np.shape(data['temperature'])[0])
    ydata = data[quantity][:,0]
    y_err = data[quantity][:,1]
    
    T_c = crit_temp(data_dir)
    
    # Fitting
    ## Selecting data for fit
    indices = np.where(((xdata-T_c)>fit_range[0]) & ((xdata-T_c)<fit_range[1]))
    xdata_fit = xdata[indices] - T_c
    ydata_fit = ydata[indices]
    y_err_fit = y_err[indices]

    ## Select fitting model
    if quantity == 'magnetisation':
        from data_processing import f_magnetisation as f
        plt.ylabel(r'$\langle \mathrm{m^2} \rangle$', fontsize=18)
    if quantity == 'c_v':
        from data_processing import f_cv as f
        plt.ylabel('$\mathrm{C_v}$', fontsize=18)
    if quantity == 'chi':
        from data_processing import f_chi as f
        plt.ylabel('$\mathrm{\chi}$', fontsize=18)
    
    if quantity == 'magnetisation':
        ## Actual fitting
        popt, pcov = curve_fit(f, abs(xdata_fit), ydata_fit, sigma=y_err_fit)
        fit_err = np.sqrt(np.diag(pcov))

        # Generate fit plot data
        xdata_fit_plot = np.linspace(xdata_fit[0], xdata_fit[-1], 1000)
        ydata_fit_plot = f(abs(xdata_fit_plot), *popt)
                
        # Plotting original function and fit
        if plotYN:

            # Actual plotting
            if LOG:
                # Actual plotting
                plt.loglog(xdata[(xdata-T_c)>0]-T_c, ydata[(xdata-T_c)>0], 'bx', markersize=7)
                plt.loglog(abs(xdata[(xdata-T_c)<0]-T_c), ydata[(xdata-T_c)<0], 'rx', markersize=7)
                plt.loglog(abs(xdata_fit_plot),ydata_fit_plot, 'k-', alpha = 0.9, label = 'Fit -')
                plt.xlabel('$\mathrm{k_B |T-T_c|/J}$')
                plt.grid()
            else:
                plt.plot(xdata[(xdata-T_c)>0]-T_c, ydata[(xdata-T_c)>0], 'bx', markersize=7)
                plt.plot((xdata[(xdata-T_c)<0]-T_c), ydata[(xdata-T_c)<0], 'rx', markersize=7)
                plt.plot((xdata_fit_plot),ydata_fit_plot, 'k-', alpha = 0.9, label = 'Fit -')
                plt.xlabel('$\mathrm{k_B (T-T_c)/J}$')
                
            plt.tight_layout()
            #plt.show()

        return popt, fit_err, xdata_fit_plot, ydata_fit_plot
        
    else:
        ## Actual fitting
        popt, pcov = curve_fit(f, abs(xdata_fit), ydata_fit, sigma=y_err_fit)
        fit_err = np.sqrt(np.diag(pcov))

        # Generate fit plot data
        xdata_fit_plot = np.linspace(xdata_fit[0], xdata_fit[-1], 1000)
        ydata_fit_plot = f(abs(xdata_fit_plot), *popt)
        
        # Plotting original function and fit
        if plotYN:
            if LOG:
                # Actual plotting
                plt.loglog(xdata[(xdata-T_c)>0]-T_c, ydata[(xdata-T_c)>0], 'bx', markersize=7)
                plt.loglog(abs(xdata[(xdata-T_c)<0]-T_c), ydata[(xdata-T_c)<0], 'rx', markersize=7)
                plt.loglog(abs(xdata_fit_plot),ydata_fit_plot, 'k-', alpha = 0.9, label = 'Fit +')
                plt.xlabel('$\mathrm{k_B |T-T_c|/J}$')
                plt.grid()
                
            else:
                plt.plot(xdata[(xdata-T_c)>0]-T_c, ydata[(xdata-T_c)>0], 'bx', markersize=7)
                plt.plot((xdata[(xdata-T_c)<0]-T_c), ydata[(xdata-T_c)<0], 'rx', markersize=7)
                plt.plot((xdata_fit_plot),ydata_fit_plot, 'k-', alpha = 0.9, label = 'Fit +')
                plt.xlabel('$\mathrm{k_B (T-T_c)/J}$')
            
            plt.tight_layout()
            #plt.show()

        return popt, fit_err, xdata_fit_plot, ydata_fit_plot

# -----------------------------------------------------------------------------------------------------------------------
# Save data
# -----------------------------------------------------------------------------------------------------------------------
        
def save_data(self, results, data_dir, identifier):
    '''Saves most imported data to a npz file
    
    Example code of how to read data from a npzfile
       {
           npzfile = np.load(sim.data_dir + 'name.npz')
           npzfile.files
           npzfile['temperature']
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