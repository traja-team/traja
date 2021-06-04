"""
Autocorrelation plotting with traja
-----------------------------------
Plot autocorrelation of a trajectory with :meth:`traja.plotting.plot_autocorrelation`

Wrapper for pandas :meth:`pandas.plotting.autocorrelation_plot`.

"""
import traja

trj = traja.generate(seed=0)
trj.traja.plot_autocorrelation('x')