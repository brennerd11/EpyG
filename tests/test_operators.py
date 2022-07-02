import pytest
import hypothesis
from hypothesis import given
from epyg import Operators, EpyG


@pytest.fixture
def state():
    return EpyG.epg()


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
    assert (T_backward * (T_forward * state)) == state


@given(shifts=hypothesis.strategies.integers(min_value=0, max_value=64))
def test_shift_and_shift_back_relaxed_state(shifts):
    state = EpyG.epg(initial_size=256)
    S_forward = Operators.Shift(shifts=shifts)
    S_back = Operators.Shift(shifts=-shifts)
    assert (S_back * (S_forward * state)) == state


@given(shifts=hypothesis.strategies.integers(min_value=0, max_value=64))
def test_shift_and_shift_back_excited_state(shifts):
    state = EpyG.epg(initial_size=256)
    T = Operators.Transform(alpha=45.0, phi=45.0)
    state = T * state
    S_forward = Operators.Shift(shifts=shifts)
    S_back = Operators.Shift(shifts=-shifts)
    assert (S_back * (S_forward * state)) == state
