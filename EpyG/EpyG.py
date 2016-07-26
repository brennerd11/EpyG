#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implementation of the Extended Phase Graph (EPG) algorithm
Very similar to (based on) Tony Stoecker's epg_step.m
Derived from Ed Prachts python implementation

@author: Daniel Brenner
"""
from __future__ import division, print_function

from collections import OrderedDict

import numpy as np
import sys
import uuid
from numpy import sin as sin
from numpy import cos as cos
from numpy import exp as exp
from numpy import abs as abs

# data type sued for epg operations - default is double complex (complex128)
DTYPE = "complex128"


class EpyG(object):
    '''
    Extendded Phase Graph (EPG) object for MR multipulse simulations.
    Transparently wraps a 3xN matrix containing the full set of Fourier coefficients representing a manifold dephased magnetisation state.
    The EPG itself has no paremeters! All parameters (T1,T2,etc.) are implemented within the Operators acting on the EPG object

    
    References
    ==========
     * Add them here.... (Scheffler, Hennig, Weigel)

    '''

    # NOTE: This is a plain straight forward implementation where the index in the state array corresponds to the state
    # NOTE: It would be much(!) more efficient for certain applications to have a sparse state matrix and/or to abstract
    #       access to the individual states to isolate this from the actual data structure at hand. This would probably
    #       conflict with the way operators are currently used!
    def __init__(self, initial_size=256, m0=1.0):
        '''
        Constructs EpyG object

        :param initial_size: initial size of the state vector
        :type initial_size: scalar, positive integer
        :param m0: initial magnetisation of Fz_0
        :type m0: scalar, double or None

        '''
        # State vector: index 0,1 transverse dephased states; index 2 longitudinal dephased states 
        self.state      = np.zeros((3 , initial_size), dtype=DTYPE)  # Should encapsulate the state and make it a property
        self.state[2, 0]= m0  # Use meq as equilibrium magnetisation; otherwise use the specified m0
        self.max_state = 0  # Maximum order of occupied states
        self.print_digits = 3  # Used in ipython notebooks to control precision/threshold for vis

    @staticmethod
    def copy(other_epg):
        '''
        Copies an existing epg
        '''
        new_epg = EpyG(initial_size=other_epg.size, m0=1.0)
        new_epg.max_state = other_epg.max_state
        new_epg.state = other_epg.state.copy()
        return new_epg

    def resize(self, new_size):
        '''
        Resizes the internal storage for the epg. Will also shrink the EPG without warning(!)

        :param new_size: new size of the state vector (number of states)
        :type new_size: scalar, positive integer
        '''
        if new_size < self.size():
            self.state = self.state[:, 0:new_size]
        else:
            # Manual resizing - the resize method of ndarray did not work as expected
            new_array = np.zeros((3, new_size), dtype=self.state.dtype)
            new_array[:, :self.max_state+1] = self.state[:, :self.max_state+1]
            self.state = new_array

        return self

    def extend(self, increment=None):
        '''
        Extends the internal state vector to a suitable (guessed) size
        '''
        if increment is None:
            new_size = self.size()*2
        else:
            new_size = self.size() + increment

        return self.resize(new_size)

    def size(self):
        '''
        Size of the EPG

        Returns
        -------
        integer giving the current size - shape of the EPG array

        '''
        return self.state.shape[1]

    def compact(self, threshold=1e-12, compact_memory=False):
        '''
        Compacts the EPG -> i.e. zeroes all states below the given threshold and reduces the max_state attribute.
        Will only reduce memory when argument compact_memory is set

        :param threshold: States below this threshold are considered "empty"
        :param compact_memory: Will reduce size of the EPG to non-zero states

        :returns:
        Nothing
        '''
        #TODO DANGER UNTESTED!

        mask = np.abs(self.state) < threshold
        mask = np.all(mask, axis=0, keepdims=False)
        self.state[:, mask] = 0.0
        self.max_state = np.max(np.argwhere(mask))

        # Make sure to remove all zero states
        if compact_memory:
            print("compacting is not well tested! Beware", file=sys.stderr)
            newstate = np.zeros((3, self.max_state+1), dtype=DTYPE)
            newstate[:, :] = self.state[0, self.max_state]
            self.state = self.max_state

        return self

    def __len__(self):
        return self.size()

    def get_state_matrix(self):
        '''
        Returns the reduced state representation as a 3xN matrix (F+,F-,Z)
        '''
        return self.state[:, 0:self.max_state+1]

    def get_order_vector(self):
        '''
        Returns a vector of integers containing the dephasing order
        '''
        return np.arange(self.max_state+1)

    def get_Z(self, order=0):
        '''
        Get the longitudinal magnetisation component k
        '''
        try:
            return self.state[2, order].copy()
        except IndexError: # Everything that is not populated is 0!
            return np.arrray(0.0, dtype=self.state.dtype)

    def get_F(self, order=0, rx_phase=0.0):
        '''
        Get the transverse magnetisation component k while being detected with
        a given receiver phase. Useful for RF spoiling
        '''
        idx = 0 if order >= 0 else 1  # Depending on +/- selects the corresponding state

        # TODO Somehow getting F does not work!!!!!!

        theta = exp(1j * rx_phase)

        try:    
            return self.state[idx, abs(order)] * theta
        except IndexError: # Everything that is not populated is 0!
            return np.arrray(0.0, dtype=DTYPE)

    def is_occupied(self, k, thresh=1e-6):
        if np.any(np.abs(self.state[:, k]) > thresh):
            return True
        else:
            return False

    def state_iterator(self, thresh=1e-6):
        '''
        Iterates over all non-empty states returning the tuple (k, [F+, F-, Z])
        '''

        for k in range(self.max_state+1):
            if self.is_occupied(k, thresh):
                yield (k, self.state[:, k])

    def plot(self, axis=None):
        import matplotlib.pylab as plt

        if axis is None:
            _, axis = plt.subplots()

        transverse = np.hstack((self.state[1, self.max_state+1:1:-1], self.state[0, 0:self.max_state+1]))
        index = np.arange(-self.max_state, self.max_state+1)

        print(index.shape)
        print(transverse.shape)
        axis.plot(index, np.abs(transverse), "o")

    def _repr_html_(self):
        """
        Tabular representation of the EPG
        Very limited. Maybe color coding the cells would be nice

        """
        cell_spec = "<td>{0:." + str(self.print_digits) + "f} </td>"
        thresh = np.power(10, -self.print_digits)

        html = ["<table>"]
        html.append("<tr>")
        html.append("<td><b>k</b></td>")

        # k row
        for state in self.state_iterator(thresh):
            html.append("<td><b>{0}</b></td>".format(state[0]))
        html.append("</tr>")
        html.append("<tr>")

        # F+ row
        html.append("<td><b>F+</b></td>")

        for state in self.state_iterator(thresh):
            html.append(cell_spec.format(state[1][0]))

        html.append("</tr>")
        html.append("<tr>")

        # F- row
        html.append("<td><b>F-</b></td>")

        for state in self.state_iterator(thresh):
            html.append(cell_spec.format(state[1][1]))

        html.append("</tr>")
        html.append("<tr>")

        # Z row
        html.append("<td><b>Z</b></td>")

        for state in self.state_iterator(thresh):
            html.append(cell_spec.format(state[1][2]))

        html.append("</tr>")
        html.append("</table>")

        return ''.join(html)

    def _repr_json_(self):
        jsonrepr = OrderedDict()
        jsonrepr['type'] = self.__class__.__name__
        jsonrepr['size'] = self.size()
        jsonrepr['k'] = range(self.max_state)
        jsonrepr['F+'] = OrderedDict()
        jsonrepr['F-'] = OrderedDict()
        jsonrepr['Z'] = OrderedDict()

        jsonrepr['F+']['real'] = self.state[0, :self.max_state].real.tolist()
        jsonrepr['F-']['real'] = self.state[1, :self.max_state].real.tolist()
        jsonrepr['Z']['real'] = self.state[2, :self.max_state].real.tolist()

        jsonrepr['F+']['imag'] = self.state[0, :self.max_state].imag.tolist()
        jsonrepr['F-']['imag'] = self.state[1, :self.max_state].imag.tolist()
        jsonrepr['Z']['imag'] = self.state[2, :self.max_state].imag.tolist()

        return jsonrepr

    # def __str__(self):
    #     spec = "{0:6." + str(self.print_digits) + "f}"
    #     thresh = np.power(10, -self.print_digits)
    #
    #     outstr = [""]
    #
    #     # k row
    #     for state in self.state_iterator(thresh):
    #         outstr.append("{0:5d}".format(state[0]))
    #     outstr.append("\n")
    #
    #     # F+ row
    #     for state in self.state_iterator(thresh):
    #         outstr.append(spec.format(state[1][0]))
    #     outstr.append("\n")
    #
    #     # F- row
    #
    #     for state in self.state_iterator(thresh):
    #         outstr.append(spec.format(state[1][1]))
    #         outstr.append("\n")
    #     # Z row
    #
    #     for state in self.state_iterator():
    #         outstr.append(spec.format(state[1][2]))
    #
    #     return ''.join(outstr)


class Operator(object):
    '''
    Base class of an operator acting on an epg object. Application of the operator will alter the EPG object!
    All derived operators should make sure to return the epg on application to allow operator chaining.

    Operators should encapsulate a abstract modification of the spin sates

    For physical simulation there shall be another layer that ties the operators together...
    '''

    def __init__(self, name=""):
        self.count = 0  # How often has this operator been called

        if name:
            self.name = name  # Optional name for operators
        else:
            self.name = str(uuid.uuid4())  # Unique names for easier possibility of serialisation

    def apply(self, epg):
        self.count += 1  # Count applications of the operator
        return epg

    def __mul__(self, other):
        # Overload multiplication just calling the apply method - intentionally rmul is not defined!
        # TODO This has to move to the apply method to have it called always
        if isinstance(other, Operator):
            return CompositeOperator(self, other)
        elif hasattr(other, "state"):
            return self.apply(other)
        else:
            raise NotImplementedError("Can not apply operator to non-EPGs")

    def __call__(self, other):
        return self.apply(other)

    def _repr_json_(self):
        reprjsondict = OrderedDict()
        reprjsondict["__type__"] = self.__class__.__name__
        reprjsondict["name"] = self.name
        reprjsondict["count"] = self.count

        return reprjsondict

    def _repr_html_(self):
        infodict = self._repr_json_()
        reprstr = ["<b>" + infodict["__type__"] + "</b>"]
        del infodict["__type__"]

        for key in infodict.keys():
            val = infodict[key]
            reprstr.append("<i>" + str(key) + "</i>" + "=" + str(val))

        return " ".join(reprstr)

    def __str__(self):
        infodict = self._repr_json_()
        reprstr = [infodict["__type__"]]
        del infodict["__type__"]

        for key in infodict.keys():
            val = infodict[key]
            reprstr.append(str(key) + "=" + str(val))

        return " ".join(reprstr)

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
        self._changed = True

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
        self._changed = True  # Recalculate transformation matrix automatically

    def __init__(self, alpha, phi, *args, **kwargs):
        super(Transform, self).__init__(*args, **kwargs)
        self._R = np.zeros((3, 3), dtype=DTYPE)
        self._alpha = alpha
        self._phi = phi
        self._changed = True

    def calc_matrix(self):
        # CHECK WITH THE ONE FROM Hargreaves... is that correct???? <<< Seems so...
        # Recaluclates the mixing matrix using the alpha and phi members

        #alpha = -self._alpha #This is required for correct rotation!
        alpha = self._alpha
        phi = self._phi
        co = cos(alpha)
        si = sin(alpha)
        ph = exp(phi * 1j)
        ph_i = 1.0 / ph
        ph2 = exp(2.0 * phi * 1j)
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

        self._changed = False

    def apply(self, epg):
        if self._changed:
            self.calc_matrix()
        epg.state = np.dot(self._R, epg.state)
        return super(Transform, self).apply(epg)  # Invoke superclass method

    def _repr_json_(self):
        reprjsondict =  super(Transform, self)._repr_json_()
        reprjsondict["alpha"] = self._alpha
        reprjsondict["phi"] = self._phi

        return reprjsondict


class PhaseIncrementedTransform(Transform):
    '''
    Phase incremented Operator to represent RF spoiling.
    Will changed it's phase each repetition
    '''

    @property
    def constant_phase_increment(self):
        '''
        RF phase will be incremented by phase_increment*i on each execution
        '''
        return self._constant_phase_increment

    @constant_phase_increment.setter
    def constant_phase_increment(self, value):
        self._constant_phase_increment = value

    @property
    def linear_phase_increment(self):
        '''
        RF phase will be incremented by phase_increment*i on each execution
        '''
        return self._linear_phase_increment

    @linear_phase_increment.setter
    def linear_phase_increment(self, value):
        self._linear_phase_increment = value

    def __init__(self, alpha, phi, linear_phase_inc = 0.0, const_phase_inc = 0.0, *args, **kwargs):
        super(PhaseIncrementedTransform, self).__init__(alpha, phi, *args, **kwargs)
        self._constant_phase_increment = const_phase_inc
        self._linear_phase_increment = linear_phase_inc

    def update(self):
        self.phi += self.count * self._linear_phase_increment
        self.phi += self.constant_phase_increment
        self.phi = np.mod(self.phi, 2.0*np.pi)  # Restrict the phase to avoid accumulation of numerical errors
        self._changed = True

    def apply(self, epg):
        self.update()
        return super(PhaseIncrementedTransform, self).apply(epg)

    def _repr_json_(self):
        reprjsondict =  super(PhaseIncrementedTransform, self)._repr_json_()
        reprjsondict["linear_phase_increment"] = self.linear_phase_increment
        reprjsondict["constant_phase_increment"] = self.constant_phase_increment
        return reprjsondict


class Epsilon(Operator):
    '''
    The "decay operator" applying relaxation and "regrowth" of the magnetisation components.
    '''
    
    def __init__(self, TR_over_T1, TR_over_T2, meq=1.0, *args, **kwargs):
        super(Epsilon, self).__init__(*args, **kwargs)
        self._E1 = exp(-TR_over_T1)
        self._E2 = exp(-TR_over_T2)
        self._meq = meq
        
    def apply(self, epg):
        epg.state[0, :] = epg.state[0, :] * self._E2  # Transverse decay
        epg.state[1, :] = epg.state[1, :] * self._E2  # Transverse decay
        epg.state[2, :] = epg.state[2, :] * self._E1  # Longitudinal decay
        epg.state[2, 0] = epg.state[2, 0] + self._meq*(1.0-self._E1)   # Regrowth of Mz
        
        return super(Epsilon, self).apply(epg)

    def _repr_json_(self):
        reprjsondict =  super(Epsilon, self)._repr_json_()
        reprjsondict["E1"] = self._E1
        reprjsondict["E2"] = self._E2
        reprjsondict["meq"] = self._meq
        return reprjsondict

        
class Shift(Operator):
    '''
    Implements the shift operator the corresponds to a 2PI dephasing of the magnetisation.
    More dephasings can be achieved by multiple applications of this operator! Currently only handles "positive" shifts

    !!! DANGER DANGER DANGER DANGER DANGER !!!

    Beware! Nothing prevents loss of the corner elements if the number of dephasings exceeds the size 
    of the state vector!  This operator is absoloutely critical and needs proper testing!

    !!! DANGER DANGER DANGER DANGER DANGER !!!

    '''

    def __init__(self, shifts=1, autogrow=True, *args, **kwargs):
        super(Shift, self).__init__(*args, **kwargs)
        self.shifts = shifts
        self._autogrow = autogrow

    def apply(self, epg):
        # Increment the state counter -> be careful here... if exception occurs max_state will have wrong value

        if self._autogrow and (epg.max_state + self.shifts) >= epg.size():
            epg.extend()

        # TODO Make multiple shifts happen without the for loop
        for i in range(np.abs(self.shifts)):
            if self.shifts > 0: # Here we handle positive shifts !!! NEED TO DOUBLE CHECK THIS !!!
                #epg.state[0,self.shifts+1:epg.max_state+self.shifts+1]   = epg.state[0, 0:epg.max_state+1] # Shift first row to the right
                epg.state[0, 1:] = epg.state[0, 0:-1]

                # Shift this one up (dephased transverse crosses zero)
                #epg.state[1,0:epg.max_state] = epg.state[1, 1:epg.max_state-1] # Shift this one left
                epg.state[1,0:-1] = epg.state[1, 1:] # Shift this one left
                epg.state[0,0 ] = np.conj(epg.state[1, 0])
            elif self.shifts < 0:  # TODO CHECK THIS PART!
                epg.state[1, 1:] = epg.state[1, 0:-1]
                epg.state[0, 0:-1] = epg.state[0, 1:]
                epg.state[1, 0] = np.conj(epg.state[0, 0])
            else:  # Else is 0 shift - do nothing
                pass

        epg.max_state += self.shifts

        return super(Shift, self).apply(epg)

    def _repr_json_(self):
        reprjsondict = super(Shift, self)._repr_json_()
        reprjsondict["shifts"] = self.shifts

        return reprjsondict
        

class Diffusion(Operator):
    '''
    Simulates diffusion effects by state dependent damping of the coefficients

    See e.g. Nehrke et al. MRM 2009 RF Spoiling for AFI paper
    '''

    def __init__(self, d, *args, **kwargs):
        '''
        :param d: dimension less diffusion damping constant - corresponding to b*D were D is diffusivity and b is "the b-value"
        :type d: scalar, floating point
        '''
        super(Diffusion, self).__init__(*args, **kwargs)
        self._d = d

    def apply(self, epg):
        '''
        Applicaton of the operator.
        Uses nomenclature from Nehrke et al.
        '''

        l = epg.get_order_vector()
        lsq = l*l
        Db1 = self._d*lsq
        Db2 = self._d*(lsq + l + 1.0/3.0)

        ED1 = exp(-Db1)
        ED2 = exp(-Db2)

        epg.state[0, :] *= ED2     # Transverse damping
        epg.state[1, :] *= ED2     # Transverse damping
        epg.state[2, :] *= ED1     # Longitudinal damping

        return super(Diffusion, self).apply(epg)

    def _repr_json_(self):
        reprjsondict = super(Diffusion, self)._repr_json_()
        reprjsondict["d"] = self._d

        return reprjsondict


class Spoil(Operator):
    '''
    Non-physical spoiling operator that zeros all transverse states
    '''

    def __init__(self, compact_states=False, *args, **kwargs):
        super(Spoil, self).__init__(*args, **kwargs)
        self.compact_states = compact_states

    def apply(self, epg):
        epg.state[0, :] = 0.0
        epg.state[1, :] = 0.0
        if self.compact_states:
            epg.compact()

        return super(Spoil, self).apply(epg)


class Observer(Operator):
    '''
    Stores EPG values - does NOT modify the EPG
    '''

    def __init__(self, F_states=(0,), Z_states=(), rx_phase=0.0, *args, **kwargs):
        super(Observer, self).__init__(*args, **kwargs)
        self._data_dict_f = OrderedDict()
        self._data_dict_z = OrderedDict()
        self.rx_phase = rx_phase  # Transverse detection phase

        for f_state in F_states:
            self._data_dict_f[f_state] = []

        for z_state in Z_states:
            self._data_dict_z[z_state] = []

    def get_F(self, order):
        '''
        Returns recorded transverse states

        Parameters
        ----------
        order: order of the state

        Returns
        -------
        Numpy array containing amplitude of the states

        '''
        try:
            return np.asarray(self._data_dict_f[order], dtype=DTYPE)
        except KeyError:
            raise KeyError("State was not recorded!")


    def get_Z(self, order):
        '''

        Return recorded Z states

        Parameters
        ----------
        order: order of the state

        Returns
        -------
        Numpy array containing amplitude of the states

        '''
        try:
            return np.asarray(self._data_dict_z[order], dtype=DTYPE)
        except KeyError:
            raise KeyError("State was not recorded!")


    def apply(self, epg):

        for f_state in self._data_dict_f.keys():
            self._data_dict_f[f_state].append(epg.get_F(order=f_state, rx_phase=self.rx_phase))

        for z_state in self._data_dict_z.keys():
            self._data_dict_z[z_state].append(epg.get_Z(order=z_state))

        return super(Observer, self).apply(epg)

    def clear(self):
        """
        Clear the internal storage

        Returns
        -------
        reference to self

        """

        for f_state in self._data_dict_f.keys():
            self._data_dict_f[f_state] = []

        for z_state in self._data_dict_z.keys():
            self._data_dict_z[z_state] = []

        return self

    def _repr_html_(self):
        cell_spec = "<td>{0:." + str(3) + "f} </td>"

        html = []
        html.append("<b>Observer</b>")
        html.append("<table>")
        for state in self._data_dict_f.keys():
            html.append("<tr>")
            html.append("<td><b>F{0}</b></td>".format(state))
            # k row
            for element in self._data_dict_f[state]:
                html.append(cell_spec.format(element))
            html.append("</tr>")
        html.append("</table>")

        html.append("<table>")
        for state in self._data_dict_z.keys():
            html.append("<tr>")
            html.append("<td><b>Z{0}</b></td>".format(state))
            # k row
            for element in self._data_dict_z[state]:
                html.append(cell_spec.format(element))
            html.append("</tr>")
        html.append("</table>")

        return "".join(html)


    def _repr_json_(self):
        reprjsondict = super(Observer, self)._repr_json_()
        reprjsondict["rx_phase"] = self.rx_phase
        reprjsondict["F"] = OrderedDict()
        reprjsondict["Z"] = OrderedDict()

        for state in self._data_dict_f.keys():
            reprjsondict["F"][state] = {"real": self.get_F(state).real.tolist(),
                                        "imag": self.get_F(state).imag.tolist()}

        for state in self._data_dict_z.keys():
            reprjsondict["Z"][state] = {"real": self.get_Z(state).real.tolist(),
                                        "imag": self.get_Z(state).imag.tolist()}

        return reprjsondict


class CompositeOperator(Operator):
    """
    Composite operator that contains several operators
    """

    def __init__(self, *args, **kwargs):
        super(CompositeOperator, self).__init__(**kwargs)
        self._operators = []

        for op in args:
            self.append(op)

    def prepend(self, operator):
        self._operators.insert(0, operator)
        return self

    def append(self, operator):
        self._operators.append(operator)
        return self

    def apply(self, epg):
        """
        Applies the composite operator to an EPG by consecutive application of the contained operators

        Parameters
        ----------
        epg to be operated on

        Returns
        -------
        epg after operator application

        """
        epg_dash = epg

        for op in reversed(self._operators):
            epg_dash = op.apply(epg_dash)

        return super(CompositeOperator, self).apply(epg_dash)

    def __mul__(self, other):
        if hasattr(other, "state"):
            return self.apply(other)
        elif isinstance(other, Operator):
            return self.append(other)
        else:
            raise NotImplementedError("Object can not be added to composite operator")

    def __rmul__(self, other):
        if isinstance(other, Operator):
            self.prepend(other)
        else:
            raise NotImplementedError("No viable multiplication for composite operator")

    def _repr_json_(self):
        reprjsondict = super(CompositeOperator, self)._repr_json_()
        reprjsondict["operators"] = OrderedDict()

        # This keeps the same ordering of the operators - applications order will be reveresed(!)
        for i, op in enumerate(self._operators):
            reprjsondict["operators"][i] = op._repr_json_()

        return reprjsondict

    def _repr_html_(self):
        infodict = self._repr_json_()
        reprstr = ["<b>" + infodict["__type__"] + "</b>"]
        del infodict["__type__"]
        del infodict["operators"]

        for key in infodict.keys():
            reprstr.append(str(key) + "=" + str(infodict[key]))

        reprstr.append("<ol>")
        for op in self._operators:
            reprstr.append("<li>{0}</li>".format(op._repr_html_()))
        reprstr.append("</ol>")

        return " ".join(reprstr)