import numpy as np
from epyg import EpyG as epyg
from epyg import operators


def test_state_comparison_equal():
    a = epyg.epg()
    b = epyg.epg()
    assert a is not b
    assert a == b


def test_state_comparison_inequal():
    T = operators.Transform(alpha=np.pi, phi=0.0)
    a = epyg.epg()
    b = T * epyg.epg()
    assert a is not b
    assert a != b


def test_size():
    a = epyg.epg(initial_size=128)
    assert a.size() == 128
    assert len(a) == a.size()


def test_extend():
    a = epyg.epg(initial_size=128)
    b = a.extend()
    assert b.size() == 256
    assert a == b
    assert a is b


def test_copy():
    a = epyg.epg()
    b = epyg.epg.copy(a)
    assert a is not b
    assert a == b
