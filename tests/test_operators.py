import hypothesis
import numpy as np
import pytest
from epyg import epyg as epyg
from epyg import operators
from hypothesis import assume, example, given


@pytest.fixture
def state():
    return epyg.epg()


def test_call_multiply():
    a = epyg.epg()
    b = epyg.epg()
    T = operators.Transform(alpha=90.0, phi=0.0)
    T * a
    T(b)
    assert a == b


def test_identity(state):
    I = operators.Identity()  # noqa: E741
    assert (I * state) == state


def test_T_with_no_flip(state):
    T = operators.Transform(alpha=0.0, phi=180.0)
    assert (T * state) == state


@given(alpha=hypothesis.strategies.floats(), phi=hypothesis.strategies.floats())
def test_neutral_flip_and_backflip(alpha, phi):
    state = epyg.epg()
    T_forward = operators.Transform(alpha=alpha, phi=phi)
    T_backward = operators.Transform(alpha=alpha, phi=-1.0 * phi)
    state2: epyg.epg = T_backward * (T_forward * state)
    print(state2.get_state_matrix())
    assert (T_backward * (T_forward * state)) == state


@given(alpha=hypothesis.strategies.floats(), phi=hypothesis.strategies.floats())
@example(alpha=np.pi / 2.0, phi=0.0)
def test_trivial_flip(alpha, phi):
    state: epyg.epg = epyg.epg(m0=1.0)
    T_forward = operators.Transform(alpha=alpha, phi=phi)
    state = T_forward * state
    magnetisation = np.abs(state.get_f())
    assume(not np.isnan(magnetisation))  # Large phase values may create a nan value
    assert magnetisation == pytest.approx(np.abs(np.sin(alpha)), 1e-8)


def test_rotation_around_x_gives_magnetsiation_on_minus_y():
    state: epyg.epg = epyg.epg(m0=1.0)
    T = operators.Transform(alpha=np.deg2rad(90.0), phi=0)
    T * state
    x = np.real(state.get_f(0))
    y = np.imag(state.get_f(0))

    assert x == pytest.approx(0.0, 1e-9)
    assert y == pytest.approx(-1.0, 1e-9)


def test_rotation_around_y_gives_magnetsiation_on_plus_x():
    state: epyg.epg = epyg.epg(m0=1.0)
    T = operators.Transform(alpha=np.deg2rad(90.0), phi=np.deg2rad(90.0))
    T * state
    x = np.real(state.get_f(0))
    y = np.imag(state.get_f(0))

    assert x == pytest.approx(1.0, 1e-9)
    assert y == pytest.approx(0.0, 1e-9)


@given(shifts=hypothesis.strategies.integers(min_value=-128, max_value=-128))
@example(shifts=1)
@example(shifts=-1)
def test_shift_and_shift_back_relaxed_state(shifts):
    state = epyg.epg(initial_size=256)
    S_forward = operators.Shift(shifts=shifts)
    S_back = operators.Shift(shifts=-shifts)
    assert (S_back * (S_forward * state)) == state


@given(shifts=hypothesis.strategies.integers(min_value=0, max_value=64))
@example(shifts=1)
def test_shift_and_shift_back_excited_state(shifts):
    state = epyg.epg(initial_size=256)
    T = operators.Transform(alpha=np.deg2rad(45.0), phi=np.deg2rad(45.0))
    state = T * state
    S_forward = operators.Shift(shifts=shifts)
    S_back = operators.Shift(shifts=-shifts)
    assert (S_back * (S_forward * state)) == state


@given(relax_factor=hypothesis.strategies.floats(min_value=0.0))
@example(relax_factor=1.0)
def test_relaxed_state_can_not_relax_further(relax_factor):
    state = epyg.epg(initial_size=256)
    E = operators.Epsilon(TR_over_T1=relax_factor, TR_over_T2=relax_factor)

    assert state == E * state


@given(relax_factor=hypothesis.strategies.floats(min_value=0.0))
def test_excited_state_will_return_to_equilibrium(relax_factor):
    state = epyg.epg(initial_size=256)
    T = operators.Transform(alpha=np.deg2rad(90.0), phi=0.0)
    E = operators.Epsilon(TR_over_T1=np.inf, TR_over_T2=np.inf)
    assert E * (T * state) == state
