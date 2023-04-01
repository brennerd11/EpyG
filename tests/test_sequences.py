import numpy as np
import pytest
from epyg import epyg as epyg
from epyg import operators
from numpy import deg2rad


def test_spin_echo_without_relaxation():
    state = epyg.epg()
    T_excite = operators.Transform(alpha=deg2rad(90.0), phi=deg2rad(0.0))
    T_refocus = operators.Transform(alpha=deg2rad(180.0), phi=deg2rad(180.0))
    Shift = operators.Shift(1)

    T_excite * state  # excite
    Shift * state  # dephase
    T_refocus * state  # refoucs

    assert np.abs(state.get_f(0)) == pytest.approx(0.0, 1e-9)
    assert np.abs(state.get_f(1)) == pytest.approx(0.0, 1e-9)
    assert np.abs(state.get_z(0)) == pytest.approx(0.0, 1e-9)

    Shift * state

    assert np.abs(state.get_f(0)) == pytest.approx(1.0, 1e-9)
    assert np.abs(state.get_z(0)) == pytest.approx(0.0, 1e-9)
    phase_echo1 = np.angle(state.get_f(0))

    Shift * state
    T_refocus * state  # refoucs
    assert np.abs(state.get_f(0)) == pytest.approx(0.0, 1e-9)
    assert np.abs(state.get_z(0)) == pytest.approx(0.0, 1e-9)

    Shift * state

    assert np.abs(state.get_f(0)) == pytest.approx(1.0, 1e-9)
    phase_echo2 = np.angle(state.get_f(0))
    assert phase_echo2 - phase_echo1 == pytest.approx(np.pi, 1e-9)
