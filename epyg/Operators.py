from __future__ import division, print_function

import json
import uuid
from collections import OrderedDict, deque

import numpy as np
from numpy import cos, exp, sin


class Operator(object):
    """
    Base class of an operator acting on an epg object. Application of the operator will alter the EPG object!
    All derived operators should make sure to return the epg on application to allow operator chaining.

    Operators should encapsulate a abstract modification of the spin sates

    For physical simulation there shall be another layer that ties the operators together...
    """

    def __init__(self, name=""):
        self.count = 0  # How often has this operator been called

        if name:
            self.name = name  # Optional name for operators
        else:
            self.name = str(
                uuid.uuid4()
            )  # Unique names for easier possibility of serialisation

    def apply(self, epg):
        self.count += 1  # Count applications of the operator
        return epg

    def reset(self):
        self.count = 0
        return self

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

    def store_to_file(self, filename, **kwargs):
        with open(filename, "w") as fp:
            json.dump(
                self._repr_json_(),
                fp,
                skipkeys=False,
                ensure_ascii=True,
                indent=4,
                encoding="utf-8",
                **kwargs
            )


class Identity(Operator):
    pass


class Transform(Operator):
    """
    Operator implementing the transformation performed by application of an RF pulse
    Assumes all inputs in radians!
    """

    DTYPE = "complex128"
    # It would be nice to have all operators "immutable" but that would make usage difficult.
    # E.g. for RF spoiling
    # The operator implements properties which will cause correct recalculation of
    # the matrix when changing the attributes

    @property
    def alpha(self):
        """
        Flipangle (in rad) of the Transform operator.
        Setting this value will cause recalculation of the internal rotation matrix.
        """
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
        """
        Phase (in rad) of the Transform operator.
        Setting this value will cause recalculation of the internal rotation matrix.
        """
        self._phi = value
        self._changed = True  # Recalculate transformation matrix automatically

    def __init__(self, alpha: float, phi: float, *args, **kwargs):
        super(Transform, self).__init__(*args, **kwargs)
        self._R = np.zeros((3, 3), dtype=Transform.DTYPE)
        self._alpha = alpha
        self._phi = phi
        self._changed = True

    def calc_matrix(self):
        # CHECK WITH THE ONE FROM Hargreaves... is that correct???? <<< Seems so...
        # Recaluclates the mixing matrix using the alpha and phi members

        # alpha = -self._alpha #This is required for correct rotation!
        alpha = self._alpha
        phi = self._phi
        co = cos(alpha)
        si = sin(alpha)
        ph = exp(phi * 1j)
        ph_i = 1.0 / ph
        ph2 = exp(2.0 * phi * 1j)
        ph2_i = 1.0 / ph2

        self._R[0, 0] = (1.0 + co) / 2.0
        self._R[0, 1] = ph2 * (1.0 - co) / 2.0
        self._R[0, 2] = si * ph * 1j

        self._R[1, 0] = ph2_i * (1.0 - co) / 2.0
        self._R[1, 1] = (1.0 + co) / 2.0
        self._R[1, 2] = -si * ph_i * 1j

        self._R[2, 0] = si * ph_i / 2.0 * 1j
        self._R[2, 1] = -si * ph / 2.0 * 1j
        self._R[2, 2] = co

        self._changed = False

    def apply(self, epg):
        if self._changed:
            self.calc_matrix()
        # epg.state = np.dot(self._R, epg.state)
        epg.state[:, 0 : epg.max_state + 1] = np.dot(  # noqa: E203
            self._R, epg.state[:, 0 : epg.max_state + 1]  # noqa: E203
        )
        return super(Transform, self).apply(epg)  # Invoke superclass method

    def _repr_json_(self):
        reprjsondict = super(Transform, self)._repr_json_()
        reprjsondict["alpha"] = self._alpha
        reprjsondict["phi"] = self._phi

        return reprjsondict


class PhaseIncrementedTransform(Transform):
    """
    Phase incremented Operator to represent RF spoiling.
    Will changed it's phase each repetition
    """

    @property
    def constant_phase_increment(self):
        """
        RF phase will be incremented by phase_increment*i on each execution
        """
        return self._constant_phase_increment

    @constant_phase_increment.setter
    def constant_phase_increment(self, value):
        self._constant_phase_increment = value

    @property
    def linear_phase_increment(self):
        """
        RF phase will be incremented by phase_increment*i on each execution
        """
        return self._linear_phase_increment

    @linear_phase_increment.setter
    def linear_phase_increment(self, value):
        self._linear_phase_increment = value

    def __init__(
        self, alpha, phi, linear_phase_inc=0.0, const_phase_inc=0.0, *args, **kwargs
    ):
        super(PhaseIncrementedTransform, self).__init__(alpha, phi, *args, **kwargs)
        self._constant_phase_increment = const_phase_inc
        self._linear_phase_increment = linear_phase_inc

    def update(self):
        self.phi += self.count * self._linear_phase_increment
        self.phi += self.constant_phase_increment
        self.phi = np.mod(
            self.phi, 2.0 * np.pi
        )  # Restrict the phase to avoid accumulation of numerical errors
        self._changed = True

    def apply(self, epg):
        self.update()
        return super(PhaseIncrementedTransform, self).apply(epg)

    def _repr_json_(self):
        reprjsondict = super(PhaseIncrementedTransform, self)._repr_json_()
        reprjsondict["linear_phase_increment"] = self.linear_phase_increment
        reprjsondict["constant_phase_increment"] = self.constant_phase_increment
        return reprjsondict


class Epsilon(Operator):
    """
    The "decay operator" applying relaxation and "regrowth" of the magnetisation components.
    """

    def __init__(self, TR_over_T1, TR_over_T2, meq=1.0, *args, **kwargs):
        super(Epsilon, self).__init__(*args, **kwargs)

        assert TR_over_T1 >= 0.0, "Tachyons?"
        assert TR_over_T2 >= 0.0, "Tachyons?"

        self._E1 = exp(-TR_over_T1)
        self._E2 = exp(-TR_over_T2)
        self._meq = meq

    def apply(self, epg):  # Somehow composite statements (*=, +=) did not work...
        epg.state[0, :] = epg.state[0, :] * self._E2  # Transverse decay
        epg.state[1, :] = epg.state[1, :] * self._E2  # Transverse decay
        epg.state[2, :] = epg.state[2, :] * self._E1  # Longitudinal decay
        epg.state[2, 0] = epg.state[2, 0] + self._meq * (
            1.0 - self._E1
        )  # Regrowth of Mz

        return super(Epsilon, self).apply(epg)

    def _repr_json_(self):
        reprjsondict = super(Epsilon, self)._repr_json_()
        reprjsondict["E1"] = self._E1
        reprjsondict["E2"] = self._E2
        reprjsondict["meq"] = self._meq
        return reprjsondict


class Shift(Operator):
    """
    Implements the shift operator the corresponds to a 2PI dephasing of the magnetisation.
    More dephasings can be achieved by multiple applications of this operator!

    !!! DANGER DANGER DANGER DANGER DANGER !!!

    Beware! Nothing prevents loss of the corner elements if the number of dephasings exceeds the size
    of the state vector!  This operator is absoloutely critical and needs proper testing!

    !!! DANGER DANGER DANGER DANGER DANGER !!!

    """

    def __init__(
        self,
        shifts: int = 1,
        autogrow: bool = True,
        compact: float = 1e-6,
        *args,
        **kwargs
    ):
        super(Shift, self).__init__(*args, **kwargs)
        self.shifts = shifts
        self._autogrow = autogrow
        self._compacting = compact

    def apply(self, epg):
        # The growing stuff should happen in the EPG not the operator
        # Increment the state counter -> be careful here... if exception occurs max_state will have wrong value
        new_max = epg.max_state + self.shifts

        # Deal with shifting exceeding the EPG size
        # TODO this must be refactored
        if new_max >= epg.size():
            if self._autogrow:
                if (
                    self._compacting
                ):  # Try to compact first and see if this resolves the issue
                    epg.compact(threshold=self._compacting)
                if (epg.max_state + self.shifts) >= epg.size():  # No? then still extend
                    epg.extend()
                new_max = epg.max_state + self.shifts
            else:
                raise IndexError(
                    "Shift is not possible as EPG size is too limited and operator does not allow autogrowing!"
                )

        # TODO Make multiple shifts happen without the for loop
        # REMARK Only shift the necessary parts of the EPG here -- does not seem to be impact speed significantly
        for i in range(np.abs(self.shifts)):
            if (
                self.shifts > 0
            ):  # Here we handle positive shifts !!! NEED TO DOUBLE CHECK THIS !!!
                # epg.state[0, 1:] = epg.state[0, 0:-1]
                # epg.state[0, 1:new_max+1] = epg.state[0, 0:new_max]

                # Shift this one up (dephased transverse crosses zero)
                # epg.state[1,0:epg.max_state] = epg.state[1, 1:epg.max_state-1] # Shift this one left
                # epg.state[1, 0:] = epg.state[1, 1:]  # Shift this one left
                # epg.state[0, 0] = np.conj(epg.state[1, 0])
                epg.state[0, 1:] = epg.state[0, 0:-1]
                epg.state[1, 0:-1] = epg.state[1, 1:]
                epg.state[0, 0] = np.conj(epg.state[1, 0])
            elif self.shifts < 0:  # TODO CHECK THIS PART!
                # epg.state[1, 1:new_max+1] = epg.state[1, 0:new_max]
                # epg.state[0, 0:new_max] = epg.state[0, 1:new_max+1]
                # epg.state[1, 0] = np.conj(epg.state[0, 0])

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
    """
    Simulates diffusion effects by state dependent damping of the coefficients

    See e.g. Nehrke et al. MRM 2009 RF Spoiling for AFI paper
    """

    def __init__(self, d, *args, **kwargs):
        """
        :param d: dimension less diffusion damping constant - corresponding to
                  b*D were D is diffusivity and b is "the b-value"
        :type d: scalar, floating point
        """
        super(Diffusion, self).__init__(*args, **kwargs)
        self._d = d

    def apply(self, epg):
        """
        Applicaton of the operator.
        Uses nomenclature from Nehrke et al.
        """

        order = epg.get_order_vector()
        order_sq = order * order
        db1 = self._d * order_sq
        db2 = self._d * (order_sq + order + 1.0 / 3.0)

        ed1 = exp(-db1)
        ed2 = exp(-db2)

        epg.state[0, :] = epg.state[0, :] * ed2  # Transverse damping
        epg.state[1, :] = epg.state[1, :] * ed2  # Transverse damping
        epg.state[2, :] = epg.state[2, :] * ed1  # Longitudinal damping

        return super(Diffusion, self).apply(epg)

    def _repr_json_(self):
        reprjsondict = super(Diffusion, self)._repr_json_()
        reprjsondict["d"] = self._d

        return reprjsondict


class Spoil(Operator):
    """
    Non-physical spoiling operator that zeros all transverse states
    """

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
    """
    Stores EPG values - does NOT modify the EPG
    """

    def __init__(self, f_states=(0,), z_states=(), rx_phase=0.0, *args, **kwargs):
        super(Observer, self).__init__(*args, **kwargs)
        self._data_dict_f = OrderedDict()
        self._data_dict_z = OrderedDict()
        self.rx_phase = rx_phase  # Transverse detection phase
        self._fstates = f_states
        self._zstates = z_states
        self._init_data_structures()

    def _init_data_structures(self):

        for f_state in self._fstates:
            self._data_dict_f[f_state] = []

        for z_state in self._zstates:
            self._data_dict_z[z_state] = []

    def get_f(self, order):
        """
        Returns recorded transverse states

        Parameters
        ----------
        order: order of the state

        Returns
        -------
        Numpy array containing amplitude of the states

        """
        try:
            return np.asarray(self._data_dict_f[order])
        except KeyError:
            raise KeyError("State was not recorded!")

    def get_z(self, order):
        """

        Return recorded Z states

        Parameters
        ----------
        order: order of the state

        Returns
        -------
        Numpy array containing amplitude of the states

        """
        try:
            return np.asarray(self._data_dict_z[order])
        except KeyError:
            raise KeyError("State was not recorded!")

    def apply(self, epg):

        for f_state in self._data_dict_f.keys():
            self._data_dict_f[f_state].append(
                epg.get_f(order=f_state, rx_phase=self.rx_phase)
            )

        for z_state in self._data_dict_z.keys():
            self._data_dict_z[z_state].append(epg.get_z(order=z_state))

        return super(Observer, self).apply(epg)

    def clear(self):
        """
        Clear the internal storage

        Returns
        -------
        reference to self

        """

        self._init_data_structures()

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
            reprjsondict["F"][state] = {
                "real": self.get_f(state).real.tolist(),
                "imag": self.get_f(state).imag.tolist(),
            }

        for state in self._data_dict_z.keys():
            reprjsondict["Z"][state] = {
                "real": self.get_z(state).real.tolist(),
                "imag": self.get_z(state).imag.tolist(),
            }

        return reprjsondict


class SteadyStateObserver(Observer):
    """
    The steady state observer will record only a specified number of last signal events.
    It further will compute the wether a steady state has been reached which can
    e.g. by used to terminate a simulation early

    """

    # TODO The steady state observer is not yet tested...
    def __init__(
        self,
        f_states=(0,),
        z_states=(),
        rx_phase=0.0,
        window_size=5,
        thresh=1e-3,
        *args,
        **kwargs
    ):
        super(SteadyStateObserver, self).__init__(
            f_states, z_states, rx_phase, *args, **kwargs
        )
        self._windows_size = window_size
        self.steady = False
        self._threshold = thresh

    def _init_data_structures(self):
        for f_state in self._fstates:
            self._data_dict_f[f_state] = deque(maxlen=self._windows_size)
        for z_state in self._zstates:
            self._data_dict_z[z_state] = deque(maxlen=self._windows_size)

    def _check_steady_state(self, signal):
        # TODO that's pretty arbitrary maybe we should simply check the variance...
        signal = np.abs(signal)
        diffmetric = np.sum(np.abs(np.diff(signal)))

        return diffmetric < self._threshold

    def _check_states(self):

        for f_state in self._fstates:
            if not self._check_steady_state(self._data_dict_f[f_state]):
                return False

        for z_state in self._zstates:
            if not self._check_steady_state(self._data_dict_f[z_state]):
                return False

        return True

    def apply(self, epg):
        ret = super(SteadyStateObserver, self).apply()
        self.steady = self._check_states()
        return ret

    def get_f(self, order):
        return super(SteadyStateObserver, self).get_f(order)[-1]

    def get_z(self, order):
        return super(SteadyStateObserver, self).get_z(order)[-1]


class FullEPGObserver(Operator):
    def __init__(self, state_size=256, transient_size=256, *args, **kwargs):
        super(FullEPGObserver, self).__init__(*args, **kwargs)
        self._state_size = state_size
        self._transient_szie = transient_size

        self._f_matrix = np.zeros((2 * state_size, transient_size))
        self._z_matrix = np.zeros((state_size, transient_size))

    def apply(self, epg):
        raise NotImplementedError("Not yet...")


class CompositeOperator(Operator):
    """
    Composite operator that contains several operators
    """

    def __init__(self, *args, **kwargs):
        super(CompositeOperator, self).__init__(**kwargs)
        self._operators = []

        for op in args:
            self.append(op)

    def select(self, name):

        for op in self._operators:
            if op.name == name:
                return op
        return None

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

        # This keeps the same ordering of the operators - applications order will be reversed(!)
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


# TODO critically consider if we need this (untested) operator
class LoopOperator(CompositeOperator):
    def __init__(self, looplength=1, *args, **kwargs):
        super(LoopOperator, self).__init__(*args, **kwargs)
        self._looplength = looplength

    def apply(self, epg):
        last = epg
        for i in range(self._looplength):
            last = super(LoopOperator, self).apply(epg)
        return last

    def set_loop_length(self, value):
        assert value > 0, "Looplength must be positive or 0!"

    def _repr_json_(self):
        reprjsondict = super(CompositeOperator, self)._repr_json_()
        reprjsondict["_looplength"] = self._looplength
        return reprjsondict
