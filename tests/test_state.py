import numpy as np
from epyg import EpyG, Operators


def test_state_comparison_equal():
    a = EpyG.epg()
    b = EpyG.epg()
    assert a is not b
    assert a == b


def test_state_comparison_inequal():
    T = Operators.Transform(alpha=np.pi, phi=0.0)
    a = EpyG.epg()
    b = T * EpyG.epg()
    assert a is not b
    assert a != b


def test_size():
    a = EpyG.epg(initial_size=128)
    assert a.size() == 128


def test_extend():
    a = EpyG.epg(initial_size=128)
    b = a.extend()
    assert b.size() == 256
    assert a == b
    assert a is b


def test_copy():
    a = EpyG.epg()
    b = EpyG.epg.copy(a)
    assert a is not b
    assert a == b
