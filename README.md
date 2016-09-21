EpyG - Extended Phase Graphs in Python
======================================
Author: Daniel Brenner, DZNE Bonn, 2016

Extended Phase Graphs (EPG) are mathematical model of the signal evolution in magnetic resonance with particular applications
imaging (MRI). 
This python implementation of the EPG formalism aim at being usable from a user perspective, being suited for educational
uses as well as providing adequate performance to allow 'real' applicaltions e.g. for more "heavy" simulations.
The overall implementation follows the Operator notation as e.g. shown by Weigel.

The code is currently works in progress! Use with great care!

## Installation
EpyG is a pure python module built upon numpy. Standard installation using
```
    python setup.py
```
should suffice therefore. Alternatively use pip to install directly from this repository

```
    pip install git+https://path.to.this.repo/EpyG
```
## Package structure
The EpyG package is structured in 3 modules
 * EpyG - the actual representation of an Extended Phase Grap
 * Operators - implementation of Operators that manipulate EpyG
 * Applications - combinations of operators and looping to simulate real life MRI problems


## First steps
For first steps it is advised to look at the example ipython/jupyter notebook(s) in the examples directory

 
## Testing
No unit testing is yet available :-( (shame on me). The example notebooks provide some basic vaildation for the scenarios
of RF spoiled GRE sequences based on the analytical signal equations