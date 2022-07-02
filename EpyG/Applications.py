from __future__ import division, print_function

import Operators as ops
import EpyG as ep
import numpy as np


def GRE_step(
    TR, T1, T2, alpha_deg, phi_deg, linear_phase_inc_deg, d=None, observe=True
):
    """
    Function the creates an CompositeOperator that mimics a traditional GRE sequence execution.
    This is a user level function to provide this often utilized building block

    Parameters
    ----------
    TR: TR in ms
    T1: T1 in ms
    T2: T2 in ms
    alpha_deg: flip angle in rad
    phi_deg: initial rf phase in rad
    linear_phase_inc: linear phase increment for RF spoiling
    observe: boolean - if yes than an observer will be added and returned
    d: Additional diffusion damping

    Returns
    -------
    c: composite operator performing the desired operation

    -- or --

    (c, o) if observe is true with o being the observer

    """
    tr = np.double(TR)
    t1 = np.double(T1)
    t2 = np.double(T2)

    T = ops.PhaseIncrementedTransform(
        alpha=np.deg2rad(alpha_deg),
        phi=np.deg2rad(phi_deg),
        linear_phase_inc=np.deg2rad(linear_phase_inc_deg),
        name="Excite",
    )

    if observe:
        O = ops.Observer(f_states=(0,), z_states=(), name="ADC_echo")
    else:
        O = ops.Observer(name="Identity")
    S = ops.Shift(shifts=1, autogrow=True, compact=1e-8, name="Spoil")
    E = ops.Epsilon(TR_over_T1=tr / t1, TR_over_T2=tr / t2, name="Relaxation")
    if d is not None:
        D = ops.Diffusion(d, name="Diffusion")
    else:
        D = ops.Identity(name="Identity")

    C = ops.CompositeOperator(D, E, S, O, T, name="GRE Propagation")

    if observe:
        return C, O
    else:
        return C


def sim_mprage(
    T1,
    T2,
    TR,
    TI,
    echo_spacing,
    echo_train_length,
    alpha_deg,
    phase_inc_deg=50.0,
    pre_loops=4,
    inversion_angle_deg=180.0,
    k_space_center=0.5,
):
    """
    Simulates an MPRAGE sequence for a given set of parameters


    Parameters
    ----------
    T1
    T2
    TR
    TI
    echo_spacing
    echo_train_length
    alpha_deg
    phase_inc_deg
    pre_loops
    inversion_angle_deg
    k_space_center: Fractional position of k-space center corresponding to TI. 0.0 = beginning (center out),
                    1.0: end

    Returns
    -------

    """

    ETL = echo_train_length
    ES = echo_spacing
    T1 = T1
    T2 = T2

    TD = TI - ETL * k_space_center
    TR = TR
    TD2 = TR - TD - ETL * ES

    # Necessary Operators
    inv = ops.Transform(
        alpha=np.deg2rad(inversion_angle_deg), phi=0.0, name="Inversion"
    )
    inv_spoil = ops.Shift(shifts=10, name="Inversion spoiling")
    td = ops.Epsilon(TR_over_T1=TD / T1, TR_over_T2=TD / T2)
    td2 = ops.Epsilon(TR_over_T1=TD2 / T1, TR_over_T2=TD2 / T2)
    c_gre, o_gre = GRE_step(
        TR=ES,
        T1=T1,
        T2=T2,
        alpha_deg=alpha_deg,
        phi_deg=0.0,
        linear_phase_inc_deg=phase_inc_deg,
        d=None,
        observe=True,
    )
    e = ep.EpyG(initial_size=ETL * 2)

    for k in range(pre_loops):
        o_gre.clear()  # We clear the observer on each iteration - just to avoid having the approach to steady state recorded
        inv * e  # Invert
        inv_spoil * e  # Spoil
        td * e  # Relax till first GRE
        for i in range(ETL):
            c_gre * e  # GRE loop
        td2 * e  # Relax till next inversion

    return o_gre.get_f(0)
