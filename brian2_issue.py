# -*- coding: utf-8 -*-
# Linear response with integral equations
# (c) May 2015 Aditya Gilra, EPFL.

"""
spiking neural net evolution with embedded weight matrix
written by Aditya Gilra (c) July 2015.
"""

from brian2 import *        # also does 'from pylab import *'
stand_alone = False
if stand_alone: set_device('cpp_standalone')
else: prefs.codegen.target = 'weave'

###### neuronal constants
R = 1.0e8*ohm
tausynE = 100.0*ms          # synaptic tau exc->exc
tausyn = 100.0*ms           # synaptic tau for all else
tau0 = 20.0*ms              # membrane tau
tau0SI = tau0/second
noise = 20.0*mV
uth = 10.0*mV
refrT = 0.5*ms

###### network constants
N = 20
NE = N/2                    # number of embedded exc neurons
NI = NE

Nbig = 100                  # total number of neurons (bgnd+embedded)
NEbig = 80

I0base = 20.0*mV/R          # base current to all neurons at all times

club = 2

####### sim constants
settletime = 1*second
stimtime = 3*second
posttime = 3*second
runtime = settletime + stimtime + posttime
tstep = defaultclock.dt
dt = tstep/second
Nsteps = int(stimtime/tstep)

deltaItimed = TimedArray(zeros(shape=(int(runtime/tstep),N*club+1))*amp,dt=tstep)

A = NEbig-NE*club
B = NEbig+NI*club
print A,B

# ###########################################
# Brian network creation
# ###########################################

# reset eta acts as a threshold increase
model_eqns = """
    dL/dt = -L/tausyn : volt
    du/dt = 1/tau0*(-u + I * R) : volt
    varI = (i>=A)*(i<B)*(i-A+1) : 1
    I = Ibase + deltaItimed( t, varI ) : amp
    Ibase : amp
"""
threshold_eqns = "u>=uth"
reset_eqns = "u=0*mV"

# the hazard function rho is the firing rate,
#  in time dt the probability to fire is rho*dt.
# noise below is only the output noise,
#  input spiking noise comes from spiking during the simulation
Nrns = NeuronGroup(Nbig, model_eqns, \
                            threshold=threshold_eqns,\
                            reset=reset_eqns,
                            refractory = refrT)
Nrns.Ibase = I0base         # constant input to all inputs
                            # there is also transient input above
Nrns.u = uniform(0.0,uth/volt,size=Nbig)*volt
                            # for LIF, u is distibuted

rates = PopulationRateMonitor(Nrns)
smbig = SpikeMonitor(Nrns)
Mu = StateMonitor(Nrns, ('varI','I','u'), record=range(Nbig), dt = tstep*1000)

run(runtime,report='text')
if stand_alone:
    device.build(directory='output', compile=True,\
                                    run=True, debug=False)

# ###########################################
# Make plots
# ###########################################

# spike raster
fig = figure()
plot(smbig.t/second, smbig.i, ',')  # all neurons
print "plotted spike raster"

figure()
for j in range(Nbig):
    plot(Mu.varI[j])
title('Timed array indices')

for j in range(80,100):
    print Mu.u[j]
print 'Some arrays of u in neurons'

show()
