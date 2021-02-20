import numpy as np
import pandas as pd
from numpy import exp
from numba import jit
from scipy.integrate import odeint
from pyDOE2 import lhs


@jit
def pituitary_ode_jit(w, t, p):
    return pituitary_ode(w, t, p)


def pituitary_ode(w, t, p):
    """
    Defines the differential equations for the pituirary gland system.
    To be used with scipy.integrate.odeint (this is the rhs equation).

    Arguments:
        w :  vector of the state variables:
                  w = [v, n, f, c]
        t :  time
        p :  vector of the parameters:
                  p = [gk, gcal, gsk, gbk, gl, k]
    """
    vca = 60
    vk = -75
    vl = -50
    Cm = 10
    vn = -5
    vm = -20
    vf = -20
    sn = 10
    sm = 12
    sf = 2
    taun = 30
    taubk = 5
    ff = 0.01
    alpha = 0.0015
    ks = 0.4
    auto = 0
    cpar = 0
    noise = 4.0

    v, n, f, c = w

    gk, gcal, gsk, gbk, gl, kc = p

    cd = (1 - auto) * c + auto * cpar

    phik = 1 / (1 + exp((vn - v) / sn))
    phif = 1 / (1 + exp((vf - v) / sf))
    phical = 1 / (1 + exp((vm - v) / sm))
    cinf = cd ** 2 / (cd ** 2 + ks ** 2)

    ica = gcal * phical * (v - vca)
    isk = gsk * cinf * (v - vk)
    ibk = gbk * f * (v - vk)
    ikdr = gk * n * (v - vk)
    ileak = gl * (v - vl)

    ikdrx = ikdr
    ibkx = ibk

    ik = isk + ibk + ikdr
    inoise = 0  # noise*w #TODO fix

    dv = -(ica + ik + inoise + ileak) / Cm
    dn = (phik - n) / taun
    df = (phif - f) / taubk
    dc = -ff * (alpha * ica + kc * c)
    return dv, dn, df, dc


def compute_pituitary_gland_df_from_parameters(downsample_rate,
                                               gcal, gsk, gk, gbk, gl, kc,
                                               sample_id,
                                               trim_start=20000,
                                               jit_compile=True):
    """
    Computes a Traja dataframe from the pituitary gland simulation.

    It is easier to discuss ion flow in term of conductances than resistances.
    If V / R = I, where V is the voltage, R is the resistance and I is the
    current, then V * C = I, where C = 1 / R is the conductance.

    Below we specify arguments in terms of maximum conductances,
    i.e. the maximum rate at which ion channels let ions through
    the cell walls.

    Arguments:
        downsample_rate : How much the dataframe will be downsampled (relative
                          to the original simulation)
        gcal            : The maximum calcium conductance
        gsk             : The maximum s-potassiun conductance
        gk              : The maximum potassium conductance
        gbk             : The maximum b-potassium conductance
        gl              : The maximum leak conductance
        kc              :
        sample_id       : The ID of this particular sample. Must be unique
        trim_start      : How much of the start of the sample to trim.
                          The start of an activation (before converging to a limit cycle
                          or fixed point) is usually not interesting from a biological
                          perspective, so the default is to remove it.
        jit_compile     : Whether to use the numba-powered just in time compiler (much
                          faster, but the numba code cannot be debugged).
    """

    # Initial conditions
    v = -60.
    n = 0.1
    f = 0.01
    c = 0.1

    p = (gk, gcal, gsk, gbk, gl, kc)
    w0 = (v, n, f, c)
    abserr = 1.0e-8
    relerr = 1.0e-6

    t = np.arange(0, 5000, 0.05)
    # print("Generating gcal={}, gsk={}, gk={}, gbk={}, gl={}, kc={}".format(gcal, gsk, gk, gbk, gl, kc))
    if jit_compile:
        wsol = odeint(pituitary_ode_jit, w0, t, args=(p,), atol=abserr, rtol=relerr)
    else:
        wsol = odeint(pituitary_ode, w0, t, args=(p,), atol=abserr, rtol=relerr)
    df = pd.DataFrame(wsol, columns=['v', 'n', 'f', 'c'])
    df = df[trim_start:]
    df['ID'] = sample_id
    df['gcal'] = gcal
    df['gsk'] = gsk
    df['gk'] = gk
    df['gbk'] = gbk
    df['gl'] = gl
    df['kc'] = kc
    df = df.iloc[::downsample_rate, :]
    # df = df.drop(columns=['t', 'ikdrx', 'ibkx'])

    return df


def create_latin_hypercube_sampled_pituitary_df(downsample_rate=100, samples=1000, jit_compile=True):
    latin_hypercube_samples = lhs(6, criterion='center', samples=samples)

    # gcal, gsk, gk, gbk, gl, kc,
    range_start = (0.5, 0.5, 0.8, 0., 0.05, 0.03)
    range_end = (3.5, 3.5, 5.6, 4., 0.35, 0.21)

    parameters = latin_hypercube_samples * range_end - latin_hypercube_samples * range_start

    dataframes = []
    for sample_id, parameter in enumerate(parameters):
        gcal, gsk, gk, gbk, gl, kc = parameter
        df = compute_pituitary_gland_df_from_parameters(downsample_rate,
                                                        gcal, gsk, gk, gbk, gl, kc,
                                                        sample_id,
                                                        jit_compile=jit_compile)
        dataframes.append(df)

    num_samples = len(dataframes)
    return pd.concat(dataframes), num_samples
