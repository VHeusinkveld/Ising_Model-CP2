# Ising Model Simulaton
## Structure
*Report/*
Contains the report.

*exported_data/*
Contains data exported from simulation runs.

*exported_figs/*
Contains figures generated from simulation runs and from more advanced 
data processing.

*Data_processing.ipynb*:
Contains all the more complex data processing that is not carried out by running 
a simulation. Among which plots from multiple data sets, fitting for critical 
exponents, and simulation performance.

*Execute_simulation.ipynb*: *MAIN FILE*
 Contains the setup of the simulation, this function should be run if a 
simulation is needed to be performed. 

*Ising_simulation.py*: 
Contains all the functions used. Among which, the metropolis and Swendsen-Wang 
algorithm.

*data_processing.py*:
Contains simple plotting and data saving functions. Also contains fitting 
functions which are used in the data_processing notebook.

*energies.py*:
Contains functions concerning energy of the system and that of single spin 
flips.

*intialisation.py*:
Contains initialisaton functions, checking input en declaring arrays.

*metropolis.py*: 
Contais functions concerning the Metropolis Monte Carlo algorithm.

*quantities.py*:
Contains all the functions to calculate the needed quantities. 

*swendsen_wang.py*
Contains functions concerning the Swendsen-Wang Monte Carlo algorithm.