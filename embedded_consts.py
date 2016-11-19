# -*- coding: utf-8 -*-

"""
Spiking neural net of LIF/SRM neurons with AI firing
written by Aditya Gilra (c) July 2015.
"""

from brian2 import *       # also does 'from pylab import *'
from rate_evolve import loadW,get_relevant_modes,get_stim_dirn

###### network constants
Nbig = 1000                 # total number of neurons (bgnd+embedded)
fexc = 0.8                  # Fraction of exc neurons
NEbig = int(fexc*Nbig)      # Number of excitatory cells
NIbig = Nbig-NEbig          # Number of inhibitory cells 

syndelay = 1*ms

####### sim constants
tstep = defaultclock.dt
dt = tstep/second

sim_type = 'evolve'
#sim_type = 'probe'

if sim_type == 'evolve':
    settletime = 2*second
    stimtime = 1*second
    posttime = 2*second
    runtime = settletime + stimtime + posttime
    Nsteps = int(stimtime/tstep)

    #init_vec_idx = -1
    init_vec_idx = 0            # first / largest response vector

    #evolve = 'EI'               # eigenvalue evolution
    evolve = 'Q'                # Hennequin et al 2014
    #evolve = None               # no embedded matrix
    #evolve = 'eye'              # identity

    #evolve_dirn = 'arb'         # arbitrary normalized initial direction
    evolve_dirn = ''            # along a0, i.e. eigvec of response energy matrix Q
    #evolve_dirn = 'eigW'        # along eigvec of W
    #evolve_dirn = 'schurW'      # along schur mode of W
    #evolve_dirn = None          # no stimulation, spontaneous evolution

    M,W,Winit,lambdas,a0s,desc_str = loadW(evolve)
    v,w,dir_str = get_relevant_modes(evolve_dirn,W,lambdas,a0s)
    y0,y01,y02 = get_stim_dirn(evolve_dirn,v,w,init_vec_idx,W)

    ###### network constants
    N = 2*M
    NE = M                     # number of embedded exc neurons
    NI = M
    I = eye(N)

    #print "Is the embedded network EI balanced?"
    #print "Ratio of exc to inh in rows of W should equal -1/gamma."
    #print sum(W[:,:M],axis=1)/sum(W[:,M:],axis=1)

    I0 = 5.0*mV               # stimulus amplitude to eigenvector
    club = 1                  # 'club' number of neurons make a rate unit

    deltaIarraysettle = zeros(shape=(int(settletime/tstep),Nbig))
    # choose one of two stim waveforms
    #stimwaveform = array([sin(2*pi*2*Hz*j*tstep) \
    #                        for j in range(Nsteps+1)])
    stimwaveform = linspace(0,1.0,Nsteps)
    deltaIarraystim = I0/mV * transpose(tile(stimwaveform,(N*club,1)))
                                # ramp input, tiled for N*club neurons
    deltaIarraystim = dot(deltaIarraystim,diag(repeat(y0,club)))
                                # stimulate neurons along y0 direction
    deltaIarraystimbig = zeros(shape=(Nsteps,Nbig))
    deltaIarraystimbig[:,NEbig-M*club:NEbig+M*club] = deltaIarraystim
    deltaIarraypost = zeros(shape=(1,Nbig))
                                # brian2 takes the last value for all times
                                #  not present in the TimedArray.
    deltaIarray = append(append(\
                                deltaIarraysettle,deltaIarraystimbig,axis=0),\
                                deltaIarraypost,axis=0)
    deltaItimed = TimedArray(deltaIarray*mV,dt=tstep)
elif sim_type == 'probe':
    settletime = 2*second
    stimtime = 1*second
    posttime = 2*second
    runtime = settletime + stimtime + posttime
    Nsteps = int(stimtime/tstep)

    #init_vec_idx = -1
    init_vec_idx = 0            # first / largest response vector

    #evolve = 'EI'               # eigenvalue evolution
    evolve = 'Q'                # Hennequin et al 2014
    #evolve = None               # no embedded matrix
    #evolve = 'eye'              # identity

    #evolve_dirn = 'arb'         # arbitrary normalized initial direction
    evolve_dirn = ''            # along a0, i.e. eigvec of response energy matrix Q
    #evolve_dirn = 'eigW'        # along eigvec of W
    #evolve_dirn = 'schurW'      # along schur mode of W
    #evolve_dirn = None          # no stimulation, spontaneous evolution

    M,W,Winit,lambdas,a0s,desc_str = loadW(evolve)
    v,w,dir_str = get_relevant_modes(evolve_dirn,W,lambdas,a0s)
    y0,y01,y02 = get_stim_dirn(evolve_dirn,v,w,init_vec_idx,W)

    ###### network constants
    N = 2*M
    NE = N/2                    # number of embedded exc neurons
    NI = NE
    I = eye(N)

    I0 = 5.0*mV                 # stimulus amplitude to eigenvector
    club = 10                   # 'club' number of neurons make a rate unit

    deltaIarraysettle = zeros(shape=(int(settletime/tstep),Nbig))
    # choose one of two stim waveforms
    #stimwaveform = array([sin(2*pi*2*Hz*j*tstep) \
    #                        for j in range(Nsteps+1)])
    stimwaveform = linspace(0,1.0,Nsteps)
    deltaIarraystim = I0/mV * transpose(tile(stimwaveform,(N*club,1)))
                                # ramp input, tiled for N*club neurons
    deltaIarraystim = dot(deltaIarraystim,diag(repeat(y0,club)))
                                # stimulate neurons along y0 direction
    deltaIarraystimbig = zeros(shape=(Nsteps,Nbig))
    deltaIarraystimbig[:,NEbig-M*club:NEbig+M*club] = deltaIarraystim
    deltaIarraypost = zeros(shape=(1,Nbig))
                                # brian2 takes the last value for all times
                                #  not present in the TimedArray.
    deltaIarray = append(append(\
                                deltaIarraysettle,deltaIarraystimbig,axis=0),\
                                deltaIarraypost,axis=0)
    deltaItimed = TimedArray(deltaIarray*mV,dt=tstep)
