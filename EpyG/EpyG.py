#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implementation of the Extended Phase Graph (EPG) algorithm
Very similar to (based on) Tony Stoecker's epg_step.m
Derived from Ed Prachts python implementation

@author: Daniel Brenner
"""
from __future__ import division

import numpy as np
from numpy import sin as sin
from numpy import cos as cos
from numpy import exp as exp
from numpy import abs as abs


class EpyG(object):
    '''
    Extendded Phase Graph (EPG) object for MR multipulse simulations.
    Transparently wraps a 3xN matrix containing the full set of Fourier coefficients representing a manifold dephased magnetisation state.
    The EPG itself has no paremeters! All parameters (T1,T2,etc.) are implemented within the Operators acting on the EPG object

    
    References
    ==========
     * Add them here.... (Scheffler, Hennig, Weigel)

    '''

    # Alternatively allow here to construct the EPG from another EPG 

    def __init__(self, meq=1.0, initial_size=256, m0 = None):
        '''
        Constructs EpyG object

        :param meq: equilibrium magnetisation
        :type meq: scalar, floating point
        :param initial_size: initial size of the state vector
        :type initial_size: scalar, psotive integer
        :param m0: initial magnetisation - either specifiy or use None to set it to meq
        :type m0: scalar, double or None

        '''
        # State vector: index 0,1 transverse dephased states; index 2 longitudinal dephased states 
        self.state     = np.zeros((3 , initial_size), dtype=complex) # Should encapsulate the state and make it a property
        self.meq       = meq
        self.state[2,0]= meq if (m0 == None) else m0 # Use meq as equilibrium magnetisation; otherwise use the specified m0
        self.max_state = 0 # Maximum order of occupied states


    def copy(self, other_epg):
        '''
        Copies an existing epg
        '''
        raise NotImplementedError("This has yet to come...")


    def resize(self, new_size):
        '''
        Resizes the internal storage for the epg. Throws exception if new size is smaller than the old one

        :param new_size: new size of the state vector
        :type new_size: scalar, positive integer
        '''
        if new_size < self.size():
            self.state = self.state[:,0:new_size]
        else:
            self.state.resize((3,new_size)) 


    def extend(self):
        '''
        Extends the internal state vector to a suitable (guessed) size
        '''
        raise NotImplementedError("Nothing here yet...")


    def size(self):
        return self.state.shape[1]


    def compact(self, threshold, compact_memory=False):
        '''
        Compacts the EPG -> i.e. zeroes all states below the given threshold and reduces the max_state attribute.
        Will only reduce memory when argument compact_memory is set
        '''
        raise NotImplementedError("This method will come very very late...")


    def __len__(self):
        return self.size()

    
    def get_state_matrix(self):
        '''
        Returns the reduced state representation as a 3xN matrix (F+,F-,Z)
        '''
        return self.state[:,0:self.max_state+1]


    def get_order_vector(self):
        '''
        Returns a vector of integers containing the dephasing order
        '''
        return np.arange(self.max_state+1)


    def get_Z(self, k=0):
        '''
        Get the longitudinal magnetisation component k
        '''
        try:
            return self.state[2,k]
        except IndexError: # Everything that is not populated is 0!
            return np.arrray(0.0, dtype=complex) # TODO Here it care should be taken to return identical datatypes as when returning a matrix element           
            

    def get_F(self, k=0, rx_phase_in_deg=0.0):
        '''
        Get the transverse magnetisation component k while being detected with
        a given receiver phase. Useful for RF spoiling
        '''
        idx = 0 if k > 0 else 1
    
        theta = exp(-1j * np.deg2rad(rx_phase_in_deg)) 
    
        try:    
            return (self.state[idx,abs(k)] * theta)
        except IndexError: # Everything that is not populated is 0!
            return np.arrray(0.0, dtype=complex)            
        
        
class Operator(object):
    '''
    Base class of an operator acting on an epg object. Application of the operator will alter the EPG object!
    All derived operators should make sure to return the epg on application to allow operator chaining.

    Operators should encapsulate a abstract modification of the spin sates - 
    e.g. for example for relaxation times only the direct exponentials are supplied.

    For physical simulation there shall be anohter layer that ties the operators together...
    '''

    def __init__(self):
        self.count = 0 # How often has this operator been called
     
    def apply(self, epg):
        self.count += 1 # Count applications of the operator
        return epg

    def __mul__(self, epg):
        # Overload multiplication just calling the apply method - intentionally rmul is not defined!
        return self.apply(epg)


class Transform(Operator):
    '''
    Operator implementing the transformation performed by application of an RF pulse
    Assumes all inputs in radians!
    '''
    
    # It would be nice to have all operators "immutable" but that would make usage difficult.
    # E.g. for RF spoiling
    # The operator implements properties which will cause correct recalculation of the matrix when changing the attributes

    @property
    def alpha(self):
        '''
        Flipangle (in rad) of the Transform operator.
        Setting this value will cause recalculation of the internal rotation matrix.
        '''
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        self.calc_matrix() # Recalculate transformation matrix automatically

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        '''
        Phase (in rad) of the Transform operator.
        Setting this value will cause recalculation of the internal rotation matrix.
        '''
        self._phi = value
        self.calc_matrix() # Recalculate transformation matrix automatically

    def __init__(self, alpha, phi):
        super(Transform, self).__init__()
        self._R = np.asmatrix(np.zeros((3, 3), dtype=complex))
        self._alpha = alpha
        self._phi = phi
        self.calc_matrix()

    def calc_matrix(self):
        # TODO CHECK WITH THE ONE FROM Hargreaves... is that correct???? <<< Seems so...
        # Recaluclates the mixing matrix using the alpha and phi members

        alpha = -self._alpha #This is required for correct rotation!
        co = cos(alpha)
        si = sin(alpha)
        ph = exp(self._phi * 1j)
        ph_i = 1.0 / ph
        ph2 = exp(2.0 * self._phi * 1j)
        ph2_i = 1.0 / ph2

        self._R[0, 0] =  (1.0 + co) / 2.0
        self._R[0, 1] =  ph2 * (1.0 - co) / 2.0
        self._R[0, 2] =  si * ph * 1j

        self._R[1, 0] =  ph2_i * (1.0 - co) / 2.0
        self._R[1, 1] =  (1.0 + co) / 2.0
        self._R[1, 2] = -si * ph_i * 1j

        self._R[2, 0] =  si * ph_i / 2.0 * 1j
        self._R[2, 1] = -si * ph / 2.0 * 1j
        self._R[2, 2] =  co
            

    def apply(self, epg):
        epg.state = self._R * epg.state  # TODO always does the full matrix-vector-mult. Only necessary on the reduced state vector!
        return super(Transform, self).apply(epg)  # Invoke superclass method
            
            
class Epsilon(Operator):
    '''
    The "decay operator" applying relaxation and "regrowth" of the magnetisation components.

    Uses directly the relaxation exponentials $E1 = exp(-T/T1)$ and $E2 = exp(-T/T2)$.
    '''
    
    def __init__(self, E1, E2):
        super(Epsilon, self).__init__()        
        self.E1 = E1 
        self.E2 = E2
        
    def apply(self, epg):
        epg.state[0:1,:] = epg.state[0:1,:] * self.E2               # Transverse decay
        epg.state[2  ,:] = epg.state[  2,:] * self.E1               # Longitudinal decay
        epg.state[2  ,0] = epg.state[  2,0] + epg.meq*(1.0-self.E1) # Regrowth of Mz
        
        return super(Epsilon, self).apply(epg)
        
        
class Shift(Operator):
    '''
    Implements the shift operator the corresponds to a 2PI dephasing of the magnetisation.
    More dephasings can be achieved by multiple applications of this operator! Currently only handles "positive" shifts

    !!! DANGER DANGER DANGER DANGER DANGER !!!

    Beware! Nothing prevents loss of the corner elements if the number of dephasings exceeds the size 
    of the state vector!  This operator is absoloutely critical and needs proper testing!

    !!! DANGER DANGER DANGER DANGER DANGER !!!

    '''

    def __init__(self, shifts=1):
        super(Shift, self).__init__()
        self.shifts = shifts
        
    def apply(self, epg):
        epg.max_state += self.shifts # TODO Increment the state counter -> be careful here... if exception occurs max_state will have wrong value

        try:
            if self.shifts > 0: # Here we handle positive shifts !!! NEED TO DOUBLE CHECK THIS !!!
                epg.state[0,self.shifts:epg.max_state]   = epg.state[0, 0:epg.max_state-self.shifts] # Shift first row to the right
                epg.state[0,0   ]              = epg.state[1, 0   ]              # Shift this one up (dephased transverse crosses zero)
                epg.state[1,0:epg.max_state-1] = epg.state[1, 1:epg.max_state-2] # Shift this one left
            else: # TODO Here we handle positive shifts - not implemented yet
                raise NotImplementedError("No negative shifts yet!")

        except ValueError:
            # Here do the following -> resize the epg and rerun the shifting - this is ok as the first command will alywas fail
            # if max_state exceeds the capacity of the vector
            raise ValueError("Shift would exceed maximum state vector size!")
        

        return super(Shift, self).apply(epg)
        

class Diffusion(Operator):
    '''
    Simulates diffusion effects by state dependent damping of the coefficients

    See e.g. Nehrke et al. MRM 2009 RF Spoiling for AFI paper
    '''

    def __init__(self, d):
        '''
        :param d: dimension less diffustion damping constant - corresponding to b*D were D is diffusivity and b is "the b-value"
        :type d: scalar, floating point
        '''
        super(Diffusion, self).__init__()
        self.d = d


    def apply(self, epg):
        '''
        Applicaton of the operator.
        Uses nomenclature from Nehrke et al.
        '''

        l = epg.get_order_vector() # Does this work??????
        lsq = l*l
        Db1 = self.d*lsq
        Db2 = self.d*(lsq + l + 1.0/3.0)

        ED1 =exp(-Db1)
        ED2 =exp(-Db2)

        epg.state[0 ,:] = epg.state[0 ,:] * ED2     # Transverse damping
        epg.state[1 ,:] = epg.state[1 ,:] * ED2     # Transverse damping
        epg.state[2 ,:] = epg.state[2 ,:] * ED1     # Longitudinal damping

        return super(Diffusion, self).apply(epg)

