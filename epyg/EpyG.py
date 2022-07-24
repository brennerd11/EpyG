#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implementation of the Extended Phase Graph (EPG) algorithm
Very similar to (based on) Tony Stoecker's epg_step.m
Derived from Ed Prachts python implementation

@author: Daniel Brenner
"""

from __future__ import annotations, division, print_function

import json
import sys
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
from numpy import abs as abs
from numpy import exp as exp

# data type sued for epg operations - default is double complex (complex128)
DTYPE = "complex128"


class epg(object):
    """
    Extendded Phase Graph (EPG) object for MR multipulse simulations.
    Transparently wraps a 3xN matrix containing the full set of Fourier coefficients representing a
    manifold dephased magnetisation state.
    The EPG itself has no paremeters!
    All parameters (T1,T2,etc.) are implemented within the Operators acting on the EPG object


    References
    ==========
     * Add them here.... (Scheffler, Hennig, Weigel)

    """

    # NOTE: This is a plain straight forward implementation where the index in the state array corresponds to the state
    # NOTE: It would be much(!) more efficient for certain applications to have a sparse state matrix and/or to abstract
    #       access to the individual states to isolate this from the actual data structure at hand. This would probably
    #       conflict with the way operators are currently used!
    def __init__(self, initial_size: int = 256, m0: float = 1.0):
        """
        Constructs EpyG object

        :param initial_size: initial size of the state vector
        :type initial_size: scalar, positive integer
        :param m0: initial magnetisation of Fz_0
        :type m0: scalar, double or None

        """
        # State vector: index 0,1 transverse dephased states; index 2 longitudinal dephased states
        self.state = np.zeros(
            (3, initial_size), dtype=DTYPE
        )  # Should encapsulate the state and make it a property
        self.state[
            2, 0
        ] = m0  # Use meq as equilibrium magnetisation; otherwise use the specified m0
        self.max_state = 0  # Maximum order of occupied states
        self.print_digits = (
            3  # Used in ipython notebooks to control precision/threshold for vis
        )

    @staticmethod
    def copy(other_epg: epg) -> epg:
        """
        Copies an existing epg
        """
        new_epg = epg(initial_size=other_epg.size(), m0=1.0)
        new_epg.max_state = other_epg.max_state
        new_epg.state = other_epg.state.copy()
        return new_epg

    def resize(self, new_size: int) -> epg:
        """
        Resizes the internal storage for the epg. Will also shrink the EPG without warning(!)

        :param new_size: new size of the state vector (number of states)
        :type new_size: scalar, positive integer
        """
        if new_size < self.size():
            self.state = self.state[:, 0:new_size]
        else:
            # Manual resizing - the resize method of ndarray did not work as expected
            new_array = np.zeros((3, new_size), dtype=self.state.dtype)
            new_array[:, : self.max_state + 1] = self.state[:, : self.max_state + 1]
            self.state = new_array

        return self

    def extend(self, increment: Optional[int] = None) -> epg:
        """
        Extends the internal state vector to a suitable (guessed) size
        """
        if increment is None:
            new_size = self.size() * 2
        else:
            new_size = self.size() + increment

        return self.resize(new_size)

    def size(self) -> int:
        """
        Size of the EPG

        Returns
        -------
        integer giving the current size - shape of the EPG array

        """
        return self.state.shape[1]

    def compact(self, threshold: float = 1e-12, compact_memory: bool = False) -> epg:
        """
        Compacts the EPG -> i.e. zeroes all states below the given threshold and reduces the max_state attribute.
        Will only reduce memory when argument compact_memory is set

        :param threshold: States below this threshold are considered "empty"
        :param compact_memory: Will reduce size of the EPG to non-zero states

        :returns:
        Nothing
        """
        # TODO DANGER UNTESTED!

        mask = np.abs(self.state) < threshold
        mask = np.all(mask, axis=0, keepdims=False)
        mask[0] = True  # Keep at least the ground state

        self.state[:, mask] = 0.0
        try:
            self.max_state = np.max(np.argwhere(np.logical_not(mask)))
        except ValueError:  # In case of a 0 mask
            self.max_state = 0

        # Make sure to remove all zero states
        if compact_memory:
            print("compacting is not well tested! Beware", file=sys.stderr)
            newstate = np.zeros((3, self.max_state + 1), dtype=DTYPE)
            newstate[:, :] = self.state[0, self.max_state]
            self.state = self.newstate

        return self

    def __len__(self) -> int:
        return self.size()

    def get_state_matrix(self) -> np.ndarray:
        """
        Returns the reduced state representation as a 3xN matrix (F+,F-,Z)
        """
        return self.state[:, 0 : self.max_state + 1]  # noqa: E203

    def get_order_vector(self) -> np.ndarray:
        """
        Returns a vector of integers containing the dephasing order
        """
        return np.arange(self.max_state + 1)

    def get_z(self, order=0) -> np.dtype:
        """
        Get the longitudinal magnetisation component k
        """
        try:
            return self.state[2, order].copy()
        except IndexError:  # Everything that is not populated is 0!
            return np.array(0.0, dtype=self.state.dtype)

    def get_f(self, order: int = 0, rx_phase: float = 0.0) -> np.dtype:
        """
        Get the transverse magnetisation component k while being detected with
        a given receiver phase. Useful for RF spoiling
        """
        idx = 0 if order >= 0 else 1  # Depending on +/- selects the corresponding state

        theta = exp(1j * rx_phase)

        try:
            return self.state[idx, abs(order)] * theta
        except IndexError:  # Everything that is not populated is 0!
            return np.array(0.0, dtype=DTYPE)

    def is_occupied(self, k: int, thresh: float = 1e-6) -> bool:
        if np.any(np.abs(self.state[:, k]) > thresh):
            return True
        else:
            return False

    def state_iterator(self, thresh: float = 1e-6):
        """
        Iterates over all non-empty states returning the tuple (k, [F+, F-, Z])
        """

        for k in range(self.max_state + 1):
            if self.is_occupied(k, thresh):
                yield (k, self.state[:, k])

    def plot(self, axis=None):
        import matplotlib.pylab as plt

        if axis is None:
            _, axis = plt.subplots()

        transverse = np.hstack(
            (
                self.state[1, self.max_state + 1 : 1 : -1],  # noqa: E203
                self.state[0, 0 : self.max_state + 1],  # noqa: E203
            )
        )
        index = np.arange(-self.max_state, self.max_state + 1)

        print(index.shape)
        print(transverse.shape)
        axis.plot(index, np.abs(transverse), "o")

    def _repr_html_(self) -> str:
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

        return "".join(html)

    def _repr_json_(self) -> Dict:
        jsonrepr = OrderedDict()
        jsonrepr["type"] = self.__class__.__name__
        jsonrepr["size"] = self.size()
        jsonrepr["k"] = range(self.max_state)
        jsonrepr["F+"] = OrderedDict()
        jsonrepr["F-"] = OrderedDict()
        jsonrepr["Z"] = OrderedDict()

        jsonrepr["F+"]["real"] = self.state[0, : self.max_state].real.tolist()
        jsonrepr["F-"]["real"] = self.state[1, : self.max_state].real.tolist()
        jsonrepr["Z"]["real"] = self.state[2, : self.max_state].real.tolist()

        jsonrepr["F+"]["imag"] = self.state[0, : self.max_state].imag.tolist()
        jsonrepr["F-"]["imag"] = self.state[1, : self.max_state].imag.tolist()
        jsonrepr["Z"]["imag"] = self.state[2, : self.max_state].imag.tolist()

        return jsonrepr

    def store_to_file(self, filename: str, **kwargs):
        with open(filename, "w") as fp:
            json.dump(
                self._repr_json_(),
                fp,
                skipkeys=False,
                ensure_ascii=True,
                indent=4,
                encoding="utf-8",
                **kwargs,
            )

    def __eq__(self, other_epg: epg) -> bool:
        if not isinstance(other_epg, epg):
            return False
        if not self.max_state == other_epg.max_state:
            return False
        return np.array_equal(self.state, other_epg.state, equal_nan=True)
