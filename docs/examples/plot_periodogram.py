"""
Periodogram plot with traja
-----------------------------------
Plot periodogram or power spectrum with :meth:`traja.plotting.plot_periodogram`.

Wrapper for pandas :meth:`scipy.signal.periodogram`.

"""
import traja

trj = traja.generate()
trj.traja.plot_periodogram('x')