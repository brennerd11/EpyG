import hypothesis
import numpy as np
import pytest
from epyg import EpyG, Operators
from hypothesis import assume, example, given
from numpy import deg2rad


def test_spin_echo_without_relaxation():
    state = EpyG.epg()
    T_excite = Operators.Transform(alpha=deg2rad(90.0), phi=deg2rad(0.0))
    T_refocus = Operators.Transform(alpha=deg2rad(180.0), phi=deg2rad(180.0))
    Shift = Operators.Shift(1)

    T_excite * state  # excite
    Shift * state  # dephase
    T_refocus * state  # refoucs

    assert np.abs(state.get_f(0)) == pytest.approx(0.0, 1e-9)

    Shift * state

    assert np.abs(state.get_f(0)) == pytest.approx(1.0, 1e-9)
    phase_echo1 = np.angle(state.get_f(0))

    Shift * state
    T_refocus * state  # refoucs
    assert np.abs(state.get_f(0)) == pytest.approx(0.0, 1e-9)

    Shift * state

    assert np.abs(state.get_f(0)) == pytest.approx(1.0, 1e-9)
    phase_echo2 = np.angle(state.get_f(0))
    assert phase_echo2 - phase_echo1 == pytest.approx(np.pi, 1e-9)
