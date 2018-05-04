import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

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