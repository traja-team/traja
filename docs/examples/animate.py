"""
Animate trajectories
-------------------------------
traja allows animating trajectories.
"""
import traja

df = traja.generate(1000, seed=0)

###############################################################################
# Plot a animation of trajectory
# ==============================
# An animation is generated using :func:`~traja.plotting.animate`.

anim = traja.plotting.animate(df) # save=True saves to 'trajectory.mp4'

####################################################################################
# .. raw:: html
#    
#    <iframe width="560" height="315" src="https://www.youtube.com/embed/-_-Ay9phB5k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>