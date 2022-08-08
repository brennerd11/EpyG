import hypothesis
import numpy as np
import pytest
from epyg import EpyG, Operators
from hypothesis import assume, example, given


@pytest.fixture
def state():
    return EpyG.epg()


def test_call_multiply():
    a = EpyG.epg()
    b = EpyG.epg()
    T = Operators.Transform(alpha=90.0, phi=0.0)
    T * a
    T(b)
    assert a == b


def test_identity(state):
    I = Operators.Identity()
    assert (I * state) == state


def test_T_with_no_flip(state):
    T = Operators.Transform(alpha=0.0, phi=180.0)
    assert (T * state) == state


@given(alpha=hypothesis.strategies.floats(), phi=hypothesis.strategies.floats())
def test_neutral_flip_and_backflip(alpha, phi):
    state = EpyG.epg()
    T_forward = Operators.Transform(alpha=alpha, phi=phi)
    T_backward = Operators.Transform(alpha=alpha, phi=-1.0 * phi)
    state2: EpyG.epg = T_backward * (T_forward * state)
    print(state2.get_state_matrix())
    assert (T_backward * (T_forward * state)) == state


@given(alpha=hypothesis.strategies.floats(), phi=hypothesis.strategies.floats())
@example(alpha=np.pi / 2.0, phi=0.0)
def test_trivial_flip(alpha, phi):
    state: EpyG.epg = EpyG.epg(m0=1.0)
    T_forward = Operators.Transform(alpha=alpha, phi=phi)
    state = T_forward * state
    magnetisation = np.abs(state.get_f())
    assume(not np.isnan(magnetisation))  # Large phase values may create a nan value
    assert magnetisation == pytest.approx(np.abs(np.sin(alpha)), 1e-8)


@given(shifts=hypothesis.strategies.integers(min_value=-128, max_value=-128))
@example(shifts=1)
@example(shifts=-1)
def test_shift_and_shift_back_relaxed_state(shifts):
    state = EpyG.epg(initial_size=256)
    S_forward = Operators.Shift(shifts=shifts)
    S_back = Operators.Shift(shifts=-shifts)
    assert (S_back * (S_forward * state)) == state


@given(shifts=hypothesis.strategies.integers(min_value=0, max_value=64))
@example(shifts=1)
def test_shift_and_shift_back_excited_state(shifts):
    state = EpyG.epg(initial_size=256)
    T = Operators.Transform(alpha=np.deg2rad(45.0), phi=np.deg2rad(45.0))
    state = T * state
    S_forward = Operators.Shift(shifts=shifts)
    S_back = Operators.Shift(shifts=-shifts)
    assert (S_back * (S_forward * state)) == state


@given(relax_factor=hypothesis.strategies.floats(min_value=0.0))
@example(relax_factor=1.0)
def test_relaxed_state_can_not_relax_further(relax_factor):
    state = EpyG.epg(initial_size=256)
    E = Operators.Epsilon(TR_over_T1=relax_factor, TR_over_T2=relax_factor)

    assert state == E * state


@given(relax_factor=hypothesis.strategies.floats(min_value=0.0))
def test_excited_state_will_return_to_equilibrium(relax_factor):
    state = EpyG.epg(initial_size=256)
    T = Operators.Transform(alpha=np.deg2rad(90.0), phi=0.0)
    E = Operators.Epsilon(TR_over_T1=np.inf, TR_over_T2=np.inf)
    assert E * (T * state) == state
