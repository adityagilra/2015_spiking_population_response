# -*- coding: utf-8 -*-

"""
Spiking neural net of LIF/SRM neurons with AI firing
written by Aditya Gilra (c) July 2015.
"""

from brian2 import *       # also does 'from pylab import *'
from embedded_consts import *

import random

## Cannot make this network a Class,
##  since brian standalone mode wants all Brian objects to be in the same scope.

###### neuronal constants
#nrn_type = 'LIF'            # Leaky Integrate-and-Fire
#nrn_type = 'SRM'            # Spike Response Model
nrn_type = 'SRM0'           # Spike Response Model exact renewal
R = 1.0e8*ohm
tausynE = 100.0*ms          # synaptic tau exc->exc
tausyn = 10.0*ms            # synaptic tau for all else
tau0 = 20.0*ms              # membrane tau
tau0SI = tau0/second
noise = 20.0*mV
uth = 10.0*mV
uth_base = 0.0*mV
refrT = 0.5*ms

###### network constants
C = 100                     # Number of incoming connections on each neuron (exc or inh)
fC = fexc                   # fraction fC incoming connections are exc, rest inhibitory
excC = int(fC*C)            # number of exc incoming connections
if nrn_type == "LIF":
    I0base = 10.5*mV/R       # base current to all neurons at all times
    J = 0.8*mV/R*(10*ms/tausynE)
else:
    I0base = 0.0*mV/R       # base current to all neurons at all times
    J = 0.8*mV/R*(10*ms/tausynE)
                            # exc strength is J (/R as we multiply by R in eqn)
                            # Critical J (for LIF network with delta synapses) is
                            #  ~ 0.45e-3 V in paper for N = 10000, C = 1000
                            # Note individual rate fluctuations
                            #  for J = 0.2e-3 V vs J = 0.8e-3 V
                            # For SRM/SRM0, synaptic filtering but no u integration
                            # In Ostojic 2014 / Brunel 2000, u integration,
                            #  but no synaptic filtering.
                            # Both are equivalent if tausyn and membrane tau are same.
                            # But LIF with synaptic filtering is different
g = 5.0*tausynE/tausyn      # if all exc syns have tausynE
#g = 5.0*(tausynE/tausyn)**2 # if only exc->exc syns have tausynE, but exc->inh is tausyn
                            # -gJ is the inh strength. For exc-inh balance g >~ f(1-f)=4
                            #  a tausynE/tausyn factor is also needed to compensate tau-s

# ###########################################
# Brian network creation
# ###########################################

# reset eta acts as a threshold increase
if nrn_type == "LIF":       # LIF
    model_eqns = """
        du/dt = 1/tau0*(-u + (Ibase + KE + K) * R + deltaItimed( t, i )) : volt
        Ibase : amp
        dKE/dt = -KE/tausynE : amp
        dK/dt = -K/tausyn : amp
    """
    threshold_eqns = "u>=uth"
    reset_eqns = "u=0*mV"
else:                       # SRM
    model_eqns = """
        u = (Ibase + KE + K) * R + deltaItimed( t, i ): volt
        Ibase : amp
        deta/dt = -eta/tau0 : volt
        dKE/dt = -KE/tausynE : amp
        dK/dt = -K/tausyn : amp
    """
    threshold_eqns = "rand()<=1.0/tau0*exp((u-(eta+uth_base))/noise)*tstep"
    if nrn_type == "SRM0": # SRM0 (exact renewal process)
        reset_eqns = "eta=uth"
    else: # usual SRM (approx as quasi-renewal process)
        reset_eqns = "eta+=uth"

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
if nrn_type == 'LIF':
    Nrns.u = uniform(0.0,uth/volt,size=Nbig)*volt
                            # for LIF, u is distibuted
else:
    Nrns.eta = uth          # initially, all SRM neurons are as if just reset

# brain2 code to make, connect and weight the background synapses
con = Synapses(Nrns,Nrns,'''w : amp
                            useSynE : 1''',\
                            pre='KE += useSynE*w; K += (1-useSynE)*w')
## Connections from some Exc/Inh neurons to each neuron
random.seed(100) # set seed for reproducibility of simulations
seed(100)
conn_i = []
conn_j = []
for jidx in range(0,Nbig):
    ## draw excC number of neuron indices out of NmaxExc neurons
    preIdxsE = random.sample(range(NEbig),excC)
    ## draw inhC=C-excC number of neuron indices out of inhibitory neurons
    preIdxsI = random.sample(range(NEbig,Nbig),C-excC)
    ## connect these presynaptically to i-th post-synaptic neuron
    ## choose the synapses object based on whether post-syn nrn is exc or inh
    conn_i += preIdxsE
    conn_j += [jidx]*excC
    conn_i += preIdxsI
    conn_j += [jidx]*(C-excC)
con.connect(conn_i,conn_j)
con.delay = syndelay
con.useSynE['i<NEbig'] = 1.0
con.w['i<NEbig'] = J
con.w['i>=NEbig'] = -g*J
#con.w = -g*J    # kind of winner take all, gives switching

