from numba import jit
from scipy.integrate import odeint

@jit
def pituitary_ode(w, t, p):
    """
    Defines the differential equations for the pituirary gland system.

    Arguments:
        w :  vector of the state variables:
                  w = [v, n, f, c]
        t :  time
        p :  vector of the parameters:
                  p = [gk, gcal, gsk, gbk, gl, k]
    """
    vca=60
    vk=-75
    vl=-50
    Cm=10
    vn=-5
    vm=-20
    vf=-20
    sn=10
    sm=12
    sf=2
    taun=30
    taubk=5
    ff=0.01
    alpha=0.0015
    ks=0.4
    auto=0
    cpar=0
    noise=4.0

    v, n, f, c = w

    gk, gcal, gsk, gbk, gl, kc = p

    cd=(1-auto)*c+auto*cpar

    phik=1/(1+exp((vn-v)/sn))
    phif=1/(1+exp((vf-v)/sf))
    phical=1/(1+exp((vm-v)/sm))
    cinf=cd**2/(cd**2+ks**2)

    ica=gcal*phical*(v-vca)
    isk=gsk*cinf*(v-vk)
    ibk=gbk*f*(v-vk)
    ikdr=gk*n*(v-vk)
    ileak=gl*(v-vl)

    ikdrx=ikdr
    ibkx =ibk

    ik = isk + ibk + ikdr
    inoise = 0 # noise*w #TODO fix

    dv = -(ica+ik+inoise+ileak)/Cm
    dn = (phik-n)/taun
    df = (phif-f)/taubk
    dc = -ff*(alpha*ica+kc*c)
    return (dv, dn, df, dc)


def compute_pituitary_gland_df_from_parameters(downsample_rate,
                                               gcal, gsk, gk, gbk, gl, kc,
                                               index):

    # Initial conditions
    v=-60.
    n=0.1
    f=0.01
    c=0.1
    
    p = (gk, gcal, gsk, gbk, gl, kc)
    w0 = (v, n, f, c)
    abserr = 1.0e-8
    relerr = 1.0e-6

    t = np.arange(0, 5000, 0.05)
    #print("Generating gcal={}, gsk={}, gk={}, gbk={}, gl={}, kc={}".format(gcal, gsk, gk, gbk, gl, kc))
    wsol = scipy.integrate.odeint(pituitary_ode, w0, t, args=(p,), atol=abserr, rtol=relerr)
    df = pd.DataFrame(wsol, columns=['v', 'n', 'f', 'c'])
    df['ID'] = index
    df['gcal'] = gcal
    df['gsk'] = gsk
    df['gk'] = gk
    df['gbk'] = gbk
    df['gl'] = gl
    df['kc'] = kc
    df = df.iloc[::downsample_rate, :]
    #df = df.drop(columns=['t', 'ikdrx', 'ibkx'])

    return df