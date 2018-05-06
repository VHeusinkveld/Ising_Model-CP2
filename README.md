# Ising Model Simulaton
## Structure
*Execute_simulation.ipynb*:
Contains the setup of the simulation, this function should be run if a 
simulation is needed to be performed. 

*Data_processing.ipynb*:
Contains all the more complex data processing that is not carried out by running 
a simulatin. Among which plots from multipla data sets, fitting for critical 
exponents, and simulation performance.

*Ising_simulation.py*: 
Contains all the functions used. Among which, the metropolis and Swendsen-Wang 
algorithm.

*data_processing.py*:
Contains plotting and data saving functions.

*energies.py*:
Contains functions concerning energy of the system and that of single spin flips.

*intialisation.py*:
Contains initialisaton functions, checking input en declaring arrays.

*metropolis.py*: 
Contais functions concerning the Metropolis Monte Carlo algorithm.

*quantities.py*:
Contains all the functions to calculate the needed quantities. 

*swendsen_wang.py*
Contains functions concerning the Swendsen-Wang Monte Carlo algorithm.