import warnings

import numpy as np
import pandas as pd
from numpy import exp
from numba import jit
from scipy.integrate import odeint
from pyDOE2 import lhs
import peakutils
from collections import OrderedDict


# PyTest will not compute coverage correctly for @jit-compiled code.
# Thus we must explicitly suppress the coverage check.
@jit
def pituitary_ode_fletcher(w, t, p):  # pragma: no cover
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


def compute_pituitary_gland_df_fletcher_from_parameters(downsample_rate,
                                                        gcal, gsk, gk, gbk, gl, kc,
                                                        sample_id,
                                                        trim_start=20000):
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
    wsol = odeint(pituitary_ode_fletcher, w0, t, args=(p,), atol=abserr, rtol=relerr)
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


def create_latin_hypercube_sampled_pituitary_df(downsample_rate=100, samples=1000):
    latin_hypercube_samples = lhs(6, criterion='center', samples=samples)

    # gcal, gsk, gk, gbk, gl, kc,
    range_start = (0.5, 0.5, 0.8, 0., 0.05, 0.03)
    range_end = (3.5, 3.5, 5.6, 4., 0.35, 0.21)

    parameters = latin_hypercube_samples * range_end - latin_hypercube_samples * range_start

    dataframes = []
    for sample_id, parameter in enumerate(parameters):
        gcal, gsk, gk, gbk, gl, kc = parameter
        df = compute_pituitary_gland_df_fletcher_from_parameters(downsample_rate,
                                                                 gcal, gsk, gk, gbk, gl, kc,
                                                                 sample_id)
        dataframes.append(df)

    num_samples = len(dataframes)
    return pd.concat(dataframes), num_samples


def uniform_scale(val_min, val_mean, val_max):
    return np.random.uniform(val_min / abs(val_mean), val_max / abs(val_mean))


@jit
def x_inf(V, V_x, s_x):
    return 1 / (1 + exp((V_x - V) / s_x))


@jit
def s_inf(c, k_s):
    return c ** 2 / (c ** 2 + k_s ** 2)


@jit
def pituitary_ode(w, t, g_CaL, g_CaT, g_K, g_SK, g_Kir, g_BK, g_NaV,
                  g_A, g_leak, C_m, E_leak, tau_m, tau_ht, tau_n,
                  tau_BK, tau_h, tau_hNa, k_c):  # pragma: no cover
    """
    Defines the differential equations for the pituirary gland system.
    To be used with scipy.integrate.odeint (this is the rhs equation).

    Arguments:
        w :  vector of the state variables:
                  w = [v, n, m, b, h, h_T, h_Na, c]
        t :  time
        p :  vector of the parameters
    """
    V, n, m, b, h, h_T, h_Na, c = w

    # (g_CaL, g_CaT, g_K, g_SK, g_Kir, g_BK, g_NaV, g_A, g_leak, C_m, E_leak,
    # tau_m, tau_ht, tau_n, tau_BK, tau_h, tau_hNa, k_c) = p

    E_Ca = 60
    E_K = -75
    # E_leak = -50 * E_leak_scale  # (-75 - -10)
    E_Na = 75

    V_m = -20
    V_mt = -38
    V_ht = -56
    V_n = -5
    V_k = -65
    V_b = -20
    V_a = -20
    V_h = -60
    V_mNa = -15
    V_hNa = -60

    # C_m = 10 * C_m_scale

    s_m = 12
    s_mt = 6
    s_ht = -5
    s_n = 10
    s_k = -8
    s_b = 2
    s_a = 10
    s_h = -5
    s_mNa = 5
    s_hNa = -10

    f_c = 0.01
    alpha = 0.0015
    k_s = 0.4  # (0.06, 0.21)

    I_CaL = g_CaL * m * (V - E_Ca)
    I_CaT = g_CaT * x_inf(V, V_mt, s_mt) * h_T * (V - E_Ca)
    I_K = g_K * n * (V - E_K)
    I_SK = g_SK * s_inf(c, k_s) * (V - E_K)
    I_Kir = g_Kir * x_inf(V, V_k, s_k) * (V - E_K)
    I_BK = g_BK * b * (V - E_K)
    I_NaV = g_NaV * pow(x_inf(V, V_mNa, s_mNa), 3) * h_Na * (V - E_Na)
    I_A = g_A * x_inf(V, V_a, s_a) * h * (V - E_K)
    I_leak = g_leak * (V - E_leak)

    I = I_CaL + I_CaT + I_K + I_SK + I_Kir + I_BK + I_NaV + I_A + I_leak

    dv = -I / C_m

    dn = (x_inf(V, V_n, s_n) - n) / tau_n
    dm = (x_inf(V, V_m, s_m) - m) / tau_m
    db = (x_inf(V, V_b, s_b) - b) / tau_BK
    dh = (x_inf(V, V_h, s_h) - h) / tau_h
    dh_T = (x_inf(V, V_ht, s_ht) - h_T) / tau_ht
    dh_Na = (x_inf(V, V_hNa, s_hNa) - h_Na) / tau_hNa

    dc = -f_c * (alpha * I_CaL + k_c * c)

    return dv, dn, dm, db, dh, dh_T, dh_Na, dc


#  OrderedDict remembers the key order. Required for python<3.9
default_ode_parameters = OrderedDict([
    ('g_CaL', 0),
    ('g_CaT', 0),
    ('g_K', 0),
    ('g_SK', 0),
    ('g_Kir', 0),
    ('g_BK', 0),
    ('g_NaV', 0),
    ('g_A', 0),
    ('g_leak', 0),
    ('C_m', 10),
    ('E_leak', -50),
    ('tau_m', 1),
    ('tau_ht', 1),
    ('tau_n', 1),
    ('tau_BK', 1),
    ('tau_h', 1),
    ('tau_hNa', 1),
    ('k_c', 0.15)]
)


def pituitary_ori_ode_parameters():
    parameters = dict()

    # Maximal conductance
    parameters['g_CaL'] = np.random.uniform(0, 4)
    parameters['g_K'] = np.random.uniform(0., 10.)
    parameters['g_leak'] = np.random.uniform(0.05, 0.4)

    # Kinetic variables
    parameters['k_c'] = np.random.uniform(0.03, 0.21)

    return parameters


def pituitary_ori_ode_parameters_Isk():
    parameters = pituitary_ori_ode_parameters()

    # Maximal conductance
    parameters['g_SK'] = np.random.uniform(.5, 3.5)

    # Other structural parameters
    parameters['C_m'] = np.random.uniform(4, 12)

    return parameters


def pituitary_ori_ode_parameters_Isk_Ibk():
    parameters = pituitary_ori_ode_parameters_Isk()

    # Maximal conductance
    parameters['g_BK'] = np.random.uniform(0, 4)

    # Kinetic variables
    parameters['tau_m'] = np.random.uniform(.7, 1.3)
    parameters['tau_n'] = np.random.uniform(20, 40)
    parameters['tau_BK'] = np.random.uniform(2, 10)

    return parameters


def pituitary_ori_ode_parameters_Isk_Ibk_Ikir():
    parameters = pituitary_ori_ode_parameters_Isk_Ibk()

    # Maximal conductance
    parameters['g_Kir'] = np.random.uniform(0, 2)

    # Other structural parameters
    parameters['E_leak'] = np.random.uniform(-75, -10)

    return parameters


def pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat():
    parameters = pituitary_ori_ode_parameters_Isk_Ibk_Ikir()

    # Maximal conductance
    parameters['g_CaT'] = np.random.uniform(0, 4)

    # Kinetic variables
    parameters['tau_ht'] = np.random.uniform(.7, 1.3)

    return parameters


def pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat_Ia():
    parameters = pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat()

    # Maximal conductance
    parameters['g_A'] = np.random.uniform(0, 100)

    parameters['tau_h'] = np.random.uniform(10, 30)

    return parameters


def pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat_Ia_Inav():
    parameters = pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat_Ia()

    # Maximal conductance
    parameters['g_NaV'] = np.random.uniform(6, 16)

    # Kinetic variables
    parameters['tau_hNa'] = np.random.uniform(1.4, 2.6)

    return parameters


def generate_pituitary(parameter_function, dt):
    # Initial conditions
    V = -60.
    n = 0.1
    m = 0.1
    b = 0.1
    h = 0.01
    h_T = 0.01
    h_Na = 0.01
    c = 0.1

    w0 = (V, n, m, b, h, h_T, h_Na, c)

    abserr = 1.0e-8
    relerr = 1.0e-6

    t = np.arange(0, 50000, dt)

    parameters = parameter_function()

    # We need to add default parameters to the simulation so
    # our ODE function executes correctly
    simulation_parameters = parameters.copy()
    for key in default_ode_parameters.keys():
        if key not in simulation_parameters:
            simulation_parameters[key] = default_ode_parameters[key]

    # Note - the key order in default_ode_parameters is correct.
    # Disregard the key order in parameters
    ode_params = tuple([simulation_parameters[key] for key in default_ode_parameters.keys()])
    wsol = odeint(pituitary_ode, w0, t, args=ode_params, atol=abserr, rtol=relerr)

    return wsol, parameters


def find_pituitary_activation_event(wsol_trimmed, V_threshold, dV_max_threshold, dV_min_threshold, dVs):
    below_event_end_threshold = np.all(np.stack((wsol_trimmed[1:] < V_threshold, dVs > dV_min_threshold), axis=-1),
                                       axis=1)
    above_event_start_threshold = np.all(np.stack((wsol_trimmed[1:] > V_threshold, dVs > dV_max_threshold), axis=-1),
                                         axis=1)
    # Skip to the end of an event
    first_event_end_index = np.argmax(below_event_end_threshold)

    if first_event_end_index >= 80000:
        return 0, 80000

    # Skip to the start of the next event
    event_start_index = np.argmax(above_event_start_threshold[first_event_end_index:]) + first_event_end_index

    if event_start_index >= 80000:
        return 0, 80000

    # Skip to the end of the next event
    event_end_index = np.argmax(below_event_end_threshold[event_start_index:]) + event_start_index

    if event_end_index >= 80000:
        return 0, 80000

    return event_start_index, event_end_index


def classify_pituitary_ode(wsol, dt, recognise_one_burst_spiking=False):
    """
    Classifies the pituitary ODE as either spiking, bursting,
    depolarised, hyperpolarised or one-spike bursting.

    The classes are returned as numbers:
    0: hyperpolarised
    1: Depolarised
    2: Spiking
    3: Bursting
    4: One-spike bursting

    Arguments:
        wsol : The sequence to classify
        recognise_one_burst_spiking : Whether to recognise one-spike bursting
                                      as separate from bursting.
    """
    min_distance = 50.

    first_10_s = round(10000. / dt)
    wsol_trimmed = wsol[first_10_s:]

    highest_peak = np.max(wsol_trimmed)
    lowest_valley = np.min(wsol_trimmed)

    min_amplitude = 10.  # 20, 30?

    if highest_peak > lowest_valley + min_amplitude:
        dVs = np.diff(wsol_trimmed)

        highest_dV = np.max(dVs)
        lowest_dV = np.min(dVs)

        V_threshold = (highest_peak - lowest_valley) * .35 + lowest_valley
        dV_max_threshold = highest_dV * .25
        dV_min_threshold = lowest_dV * .25

        try:
            top_peaks = peakutils.indexes(wsol_trimmed, thres=0.8)
        except ValueError:
            # A small number of sequences generate ValueErrors.
            # Since there are only a few, we will suppress it.
            top_peaks = []
        if len(top_peaks) > 1:
            stride = max(round((top_peaks[1] - top_peaks[0]) / 50), 1)
        else:
            stride = 1

        event_start, event_end = find_pituitary_activation_event(wsol_trimmed, V_threshold, dV_max_threshold,
                                                                 dV_min_threshold, dVs)
        # Error handling
        if event_end >= 80000:
            return 0, (20000, 21000)

        min_dist = round((event_end - event_start) * 0.05)
        if min_dist < min_distance:
            min_dist = min_distance

        nearby_peaks = peakutils.indexes(wsol_trimmed[event_start:event_end], thres=0.3, min_dist=min_dist)

        if len(nearby_peaks) > 1:
            return 3, (event_start, event_end)

        else:
            event_amplitude = np.max(wsol_trimmed[event_start:event_end]) - np.min(wsol_trimmed[event_start:event_end])

            V_area = np.sum(wsol_trimmed[event_start:event_end]) - V_threshold * (event_end - event_start)
            if V_area > 3000 / dt and event_amplitude > 30:
                if recognise_one_burst_spiking:
                    return 4, (event_start, event_end)
                else:
                    return 3, (event_start, event_end)
            else:
                return 2, (event_start, event_end)

    else:
        if highest_peak > -30:
            return 1, (20000, 21000)
        else:
            return 0, (20000, 21000)


def generate_pitutary_dataframe(parameter_function, sample_id: int, trim_start: int, downsample_rate: int,
                                classify: bool, recognise_one_burst_spiking: bool, retain_trajectories: bool,
                                add_timesteps: bool):
    if not classify and not retain_trajectories:
        raise ValueError("Error! Generated samples will have no class and no trajectory!")

    dt = 0.5
    pituitary_simulation, parameters = generate_pituitary(parameter_function, dt)
    if retain_trajectories:
        df = pd.DataFrame(pituitary_simulation, columns=['V', 'n', 'm', 'b', 'h', 'h_T', 'h_Na', 'c'])
    else:
        df = pd.DataFrame()
    if add_timesteps:
        df['timesteps'] = np.arange(0, 50000, dt)
    if classify:
        voltage_simulation = pituitary_simulation[:, 0]
        simulation_class, (_, _) = classify_pituitary_ode(voltage_simulation, dt,
                                                          recognise_one_burst_spiking=recognise_one_burst_spiking)
        df['class'] = simulation_class
        if retain_trajectories:
            df['class'] = simulation_class
        else:
            df['class'] = [simulation_class]
    if retain_trajectories:
        df = df[trim_start:]
        df['ID'] = sample_id
        for key, value in parameters.items():
            df[key] = value

        df = df.iloc[::downsample_rate, :]
    else:
        df['ID'] = [sample_id]
        for key, value in parameters.items():
            df[key] = [value]

    return df


def generate_pituitary_dataset(parameter_function, num_samples, trim_start: int = 20000, downsample_rate: int = 20,
                               classify: bool = False, recognise_one_burst_spiking: bool = False,
                               retain_trajectories: bool = False, add_timesteps: bool = False):
    """
    Computes a dataset of Traja dataframes representing
    pituitary gland simulations. The parameters are taken
    from Fletcher et al (2016) and slightly modified.

    To run this function, provide one of the parameter
    generating functions to create a dictionary of
    parameters. The full list is provided by the
    default_ode_parameters ordered dictionary.

    The parameter functions are of the format:
     * pituitary_ori_ode_parameters
    to
     * pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat_Ia_Inav
    with one ion channel (Ixx) being added in each step.

    Arguments:
        parameter_function :  Function generating a parameter
            dictionary
        num_samples        :  The number of samples to generate
        trim_start         :  How many samples to trim at the start
            of the sequence. Default is 20,000
        downsample_rate    :  The downsampling factor applied to each
            time series. Default is 20, meaning that there are
            100,000/20 = 5,000 steps in each sample.
        classify           :  Whether to classify the sequence as
            spiking, bursting, one-spike bursting (see next option),
            nonexcitable or depolarised. This operation is expensive
            and therefore disabled by default.
        recognise_one_burst_spiking : Whether to recognise one-spike
            bursting as a separate class or consider it a type
            of bursting. Disabling reduces the number of classes
            from 5 to 4. Usually one-spike bursting is less interesting
            when there are more ion channels.
        retain_trajectories : Whether to retain the trajectories in the
            dataframe. The dataframe will be significantly larger.
            Defaults to false.
        add_timesteps : Whether to add timesteps (in the column 'timesteps')
            in the dataframe. Defaults to false.
    """
    if recognise_one_burst_spiking and not classify:
        warnings.warn("Classification not requested but a classification option is set." +
                      "This is likely a mistake - please check the training options")

    dataframes = list()
    for sample_id in range(num_samples):
        df = generate_pitutary_dataframe(parameter_function, sample_id=sample_id, trim_start=trim_start,
                                         downsample_rate=downsample_rate, classify=classify,
                                         recognise_one_burst_spiking=recognise_one_burst_spiking,
                                         retain_trajectories=retain_trajectories, add_timesteps=add_timesteps)
        dataframes.append(df)

    return pd.concat(dataframes)
