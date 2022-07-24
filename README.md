EpyG - Extended Phase Graphs in Python
======================================
Author: Daniel Brenner, (formerly) DZNE Bonn, 
Started in: 2016, continued in 2022

Extended Phase Graphs (EPG) are mathematical model of the signal evolution in magnetic resonance with particular applications
imaging (MRI). 
This python implementation of the EPG formalism aim at being usable from a user perspective, being suited for educational
uses as well as providing adequate performance to allow 'real' applicaltions e.g. for more "heavy" simulations.
The overall implementation follows the Operator notation as e.g. shown by Weigel.

The code is currently works in progress! Use with great care!

## Installation
EpyG is a pure python module built upon numpy.

it uses [poetry](https://python-poetry.org) as build and distribution tool.

## Package structure
The EpyG package is structured in 3 modules
 * EpyG - the actual representation of an Extended Phase Grap
 * Operators - implementation of Operators that manipulate EpyG
 * Applications - combinations of operators and looping to simulate real life MRI problems


## First steps
For first steps it is advised to look at the example ipython/jupyter notebook(s) in the examples directory

 
## Testing
The folder tests/ contains a set of unit/integration tests to prove basic functionality.
Further application examples can be found in the examples subfolder where Jupyter notebooks illustrating the major concepts are distributed

