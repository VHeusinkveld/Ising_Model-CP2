import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -----------------------------------------------------------------------------------------------------------------------
# Plot functions
# -----------------------------------------------------------------------------------------------------------------------
def grid_plot(self, results):
    '''Gives plot of the spins in the grid
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    results : NameSpace
        contains all the simulation results        
    '''
    save = self.save_fig
    figure_directory = self.fig_dir
    
    x = results.grid_coordinates[0]
    y = results.grid_coordinates[1]
    S = results.grid_spins
        
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=16)
    
    image = plt.imshow(S, extent=(x.min(), x.max(), y.max(), y.min()), interpolation='nearest', cmap=cm.plasma)
    plt.clim(-1,1)
    plt.xlabel('x')
    plt.ylabel('y')
    if save:
        plt.savefig(figure_directory + 'grid_spins.png')
    plt.close()
    

def visualize_islands(self, results):
    '''Gives plot of the clusters in the grid
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    results : NameSpace
        contains all the simulation results        
    '''
    if self.algorithm == 'SW':
        save = self.save_fig
        figure_directory = self.fig_dir

        islands = results.islands

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('font', size=16)

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
        plt.xlabel('x')
        plt.ylabel('y')

        if save:
            plt.savefig(figure_directory + 'clusters.png')
        plt.close()
    
def plot_func(self, results):
    '''Plots all required quantities
    
    Parameters
    ----------
    self : NameSpace
        contains all the simulation parameters
    results : NameSpace
        contains all the simulation results        
    '''
    
    save = self.save_fig
    figure_directory = self.fig_dir
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=16)
    
    x = results.temperature
    y = results.c_v[:,0]
    y_err = results.c_v[:,1]
    
    plt.errorbar(x, y, yerr=y_err, fmt='x', markersize=6, capsize=4)
    plt.xlabel('$\mathrm{k_b T/J}$', fontsize=18)
    plt.ylabel('$\mathrm{C_v}$', fontsize=18)
    plt.tight_layout()
    if save:
        plt.savefig(figure_directory + self.algorithm + str(self.eq_data_points) + '_cv.png')
    plt.close()

    x = results.temperature
    y = results.chi[:,0]
    
    plt.plot(x, y, 'o')
    plt.xlabel('$\mathrm{k_b T/J}$', fontsize=18)
    plt.ylabel('$\chi$', fontsize=18)
    plt.tight_layout()
    if save:
        plt.savefig(figure_directory + self.algorithm + str(self.eq_data_points) + '_chi.png')
    
    plt.show()
    plt.close()
    
    x = results.temperature
    y = results.magnetisation[:,0]
    y_err = results.magnetisation[:,1]

    plt.errorbar(x, y, yerr=y_err, fmt='x', markersize=6, capsize=4)
    plt.xlabel('$\mathrm{k_b T/J}$', fontsize=18)
    plt.ylabel('m', fontsize=18)    
    plt.tight_layout()
    if save:
        plt.savefig(figure_directory + self.algorithm + str(self.eq_data_points) + '_m.png')
    plt.close()
    