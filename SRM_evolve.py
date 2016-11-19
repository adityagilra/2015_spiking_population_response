# -*- coding: utf-8 -*-
# Linear response with integral equations
# (c) May 2015 Aditya Gilra, EPFL.

"""
rate units evolution with Hennequin matrix
written by Aditya Gilra (c) May 2015.
"""

from brian2 import *    # also does 'from pylab import *'
stand_alone = True
if stand_alone: set_device('cpp_standalone')
else: prefs.codegen.target = 'weave'

import pickle
from scipy.integrate import odeint

sys.path.insert(0, '..')
from neurtheor.utils import *

rndseed = 108

#evolve = 'EI'              # eigenvalue evolution
evolve = 'Q'                # Hennequin et al 2014

#evolve_dirn = 'arb'        # arbitrary normalized initial direction
evolve_dirn = ''            # along a0

if evolve == 'EI':
    filestart = 'eigenW'
    M = 10                  # number of E neurons = number of I neurons
    rndseed = 106
    WisNormal = False       # decide if normal or non-normal W is to be loaded
    W,lambdas,a0s = pickle.load( open( 
                                    filestart+str(rndseed)+"M"+str(M)+\
                                        'normal'+str(WisNormal)+".pickle",
                                "rb" ) )
    if WisNormal:
        desc_str = 'real normal W'
    else:
        desc_str = 'real non-normal (EI) stable W'
else:
    filestart = 'stabW'
    M = 10                  # number of E neurons = number of I neurons
    rndseed = 100
    W,Winit,lambdas,a0s = pickle.load( open( 
                                    filestart+str(rndseed)+"M"+str(M)+".pickle",
                                "rb" ) )
    desc_str = 'stabilized SOC (EI)'

N = 2*M
I = eye(N)

if evolve_dirn == '':
    a0 = real(a0s[0])       # the initial input vector
                            # eigenvector of Q or W
                            # if cc pair of eigenvalues,
                            #  real(eigvec) stim gives exp()*cos() response
                            # if single real eigenvalue,
                            #  real(eigvec) stim gives exp() response
else:
    a0 = uniform(-1,1,N)    # random initial direction
    a0 /= norm(a0)          # normalized

# neuronal constants
SRM0 = True
R = 1.0e8*ohm
tausyn = 100.0*ms
tau0 = 20.0*ms
tau0SI = tau0/second
noise = 20.0*mV
uth = 10.0*mV
uth_base = 10.0*mV
refrT = 0.5*ms

# network constants
NE = N/2                    # number of exc neurons
p = 0.1
if evolve=='EI': w0 = 100*mV/p/N/R
else: w0 = 100*mV/p/N/R*(10*ms/tausyn)  # compensate for tausyn
                            # to keep integral of voltage input constant  
I0 = 100.0*mV/R             # stimulus amplitude to eigenvector
I0base = 0.0*mV/R           # base current to all neurons at all times
syndelay = 1*ms

# sim constants
settletime = 1*second
stimtime = 2*second
posttime = 1*second
runtime = settletime + stimtime + posttime
tstep = defaultclock.dt
dt = tstep/second
Nsteps = int(stimtime/tstep)

deltaIarraysettle = zeros(shape=(int(settletime/tstep),N))
## choose one of two stim waveforms
#stimwaveform = array([sin(2*pi*2*Hz*j*tstep) \
#                        for j in range(Nsteps+1)])
stimwaveform = linspace(0,1.0,Nsteps)
deltaIarraystim = I0/amp * transpose(tile(stimwaveform,(N,1)))
                            # ramp input, tiled for N neurons
deltaIarraystim = dot(deltaIarraystim,diag(a0))
                            # stimulate neurons along a0 direction
#deltaIarraypost = zeros(shape=(int(posttime/tstep),N))
deltaIarraypost = zeros(shape=(1,N))
deltaIarray = append(append(\
                        deltaIarraysettle,deltaIarraystim,axis=0),\
                        deltaIarraypost,axis=0) * amp
deltaItimed = TimedArray(deltaIarray,dt=tstep)
                            # no entries for posttime makes brian2 
                            #  take the last value for all future times!
# reset eta acts as a threshold increase
model_eqns = """
    u = (I + K) * R : volt
    I = Ibase + deltaItimed(t,i) : amp
    Ibase : amp
    deta/dt = -eta/tau0 : volt
    dK/dt = -K/tausyn : amp
"""
threshold_eqns = "rand()<=1.0/tau0*exp((u-(eta+uth_base))/noise)*tstep"
if SRM0: # SRM0 (exact renewal process)
    reset_eqns = "eta=uth"
else: # usual SRM (approx as quasi-renewal process)
    reset_eqns = "eta+=uth"

# the hazard function rho is the firing rate,
#  in time dt the probability to fire is rho*dt.
# noise below is only the output noise,
#  input spiking noise comes from spiking during the simulation
Nrns = NeuronGroup(N, model_eqns, \
                    threshold=threshold_eqns,\
                    reset=reset_eqns,
                    refractory = refrT)
Nrns.Ibase = I0base         # constant input to all inputs
                            # there is also transient input above
Nrns.eta = uth              # initially, all neurons are as if just reset
NrnsE = Nrns[:NE]
NrnsI = Nrns[NE:]

# brian2 code to make, connect, weight synapses
syns = Synapses(Nrns, Nrns, 'w : amp', pre='K += w') # E to E
syns.connect(True)

# how to get the -I for spiking neurons?
syns.w = w0*transpose(W).flatten()
                            # flattening weights matrix to 1D as per i*N+j
                            # brian2 also converts from [i,j] to 1D similarly
                            # BUT brian2 uses pre x post convention
                            #   while W is post x pre, hence transpose.
syns.delay = syndelay

rates = PopulationRateMonitor(Nrns)
sm = SpikeMonitor(Nrns)
Mu = StateMonitor(Nrns, 'u', record=append(range(5),range(NE,NE+5)))

run(runtime,report='text')
if stand_alone:
    device.build(directory='output', compile=True,\
                                    run=True, debug=False)

# always convert spikemon.t and spikemon.i to array-s before indexing
# spikemon.i[] indexing is extremely slow!
spiket = array(sm.t/second) # take spiketimes of all neurons
spikei = array(sm.i)

# ###########################################
# Make plots
# ###########################################

# spike raster
figure()
plot(sm.t/second, sm.i, '.')

print "plotted spike raster"

# individual rates
ratefig = figure()
rateax = ratefig.add_subplot(111)
binunits = 100
bindt = tstep*binunits
bins = range(int(runtime/bindt))
Nbins = len(bins)

timeseries = arange(0,runtime/second+dt,dt)*1000 # ms
poprate = zeros(len(timeseries))
a0rate = zeros(len(timeseries))
for j in range(N):
    ratei = rate_from_spiketrain(spiket,spikei,runtime/second,j,25e-3)
    poprate[:len(ratei)] += ratei       # need to /N for rate per neuron
    a0rate[:len(ratei)] += a0[j]*ratei  # a0 is normalized, so no /N
    if j<10:
        rateax.plot(timeseries[:len(ratei)],ratei)
poprate /= N

fig = figure()
#ax = fig.add_subplot(111)
ax = fig.add_axes([0.05, 0.1, 0.8, 0.8])
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
cutoff = max(amax(W),abs(amin(W)))
im = ax.matshow(W,cmap=cm.coolwarm,vmin=-cutoff,vmax=cutoff)
fig.colorbar(im,cax=cax)

figure()
#plot(rates.t/second,rates.rate/Hz)
plot(timeseries,poprate,label='poprate')
plot(timeseries,a0rate,label="a0rate")
legend()
xlabel('time (s)')
ylabel('pop rate (Hz)')

figure()
for j in range(10):
    plot(Mu.t/second,Mu.u[j]/mV,['r','b'][j/5])
xlabel('time (s)')
ylabel('u (mV)')

show()
