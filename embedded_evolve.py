# -*- coding: utf-8 -*-
# Linear response with integral equations
# (c) May 2015 Aditya Gilra, EPFL.

"""
spiking neural net evolution with embedded weight matrix
written by Aditya Gilra (c) July 2015.
"""

import sys
sys.path.insert(0, '..')
from neurtheor.utils import *

from brian2 import *        # also does 'from pylab import *'
stand_alone = True
if stand_alone: set_device('cpp_standalone')
else: prefs.codegen.target = 'weave'

from ExcInhNetflex import * # imports the ExcInh AI network

import pickle
import random
from scipy.integrate import odeint

if evolve is None or evolve=='eye':
    club_wt_ratio = 0.1     # extra factor of within club-assembly weights
    #w0 = J/club
    w0 = 200*mV/R*(10*ms/tausynE)/club
else:
    club_wt_ratio = 1.0     # extra factor of within club-assembly weights
                            # should not have it different from 1 to compare with rate model
    w0 = 20*mV/R*(10*ms/tausynE)/club
                            # compensate for tausyn
                            # compensate for all to all connections between
                            #  neurons in various rate units to keep
                            #  total voltage input the same for different club

print "The perturbed weight is",w0,", while bgnd weight is",J
print "The input to embedded neurons is multiplied by",I0
print "The number of neurons clubbed into a rate unit =",club
print "Within group neuron weights are multiplied by",club_wt_ratio

NrnsEmbed = Nrns[NEbig-NE*club:NEbig+NI*club]
NrnsE = NrnsEmbed[:NE*club]
NrnsI = NrnsEmbed[NE*club:]

# brian2 code to make, connect, weight the embedded dsynapses
syns = Synapses(NrnsEmbed, NrnsEmbed,\
                            '''w : amp
                            useSynE : 1''',\
                            pre='KE += useSynE*w; K += (1-useSynE)*w')
syns.connect(True)
con.useSynE['i<NE*club'] = 1.0
# how to get the -I in (W-I) for spiking neurons:
# comes automatically due to membrane decay?
Wmod = copy(W)
for idx in range(N):
    Wmod[idx,idx] *= club_wt_ratio
                            # clubbed neurons have modified weights within
Wrepeated = repeat(repeat(Wmod,club,axis=0),club,axis=1)
                            # flattening weights matrix to 1D as per i*N+j
                            # brian2 also converts from [i,j] to 1D similarly
                            # BUT brian2 uses pre x post convention
                            #   while W is post x pre, hence transpose.
syns.w = w0*transpose(Wrepeated).flatten()
                            # repeat array elements club times in both dimensions
syns.delay = syndelay

rates = PopulationRateMonitor(Nrns)
smbig = SpikeMonitor(Nrns)
sm = SpikeMonitor(NrnsEmbed)
Mu = StateMonitor(NrnsEmbed, 'u', record=append(range(5),range(NE,NE+5)))

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
fig = figure()
subplot(211)
plot(sm.t/second, sm.i, ',')        # only embedded neurons
subplot(212)
plot(smbig.t/second, smbig.i, ',')  # all neurons

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
rates = []
for j in range(0,N*club,club):
    ## sm records spikes from only embedded neurons
    ## average over club number of neurons as a rate unit
    ratei = rate_from_spiketrains(spiket,spikei,runtime/second,dt,range(j,j+club))
    rates.append(ratei)
    poprate[:len(ratei)] += ratei       # need to /N for rate per neuron
    a0rate[:len(ratei)] += y0[j/club]*ratei  # a0 is normalized, so no /N
    rateax.plot(timeseries[:len(ratei)],ratei)
poprate /= N

print "plotted individual rates"

def matevolve(y,t):
    return dot((W-I)/tau0SI,y)
trange = arange(0.0,posttime/second,0.0001)
y = odeint(matevolve,y0,trange)

figure()
for j in range(0,N):
    meanrate = mean(rates[j][int(settletime/tstep)-1000:int(settletime/tstep)])
    plot(timeseries[:len(ratei)],rates[j]-meanrate,linestyle='solid')
twinx()
for j in range(0,N):
    plot(trange*1000+(settletime+stimtime)/ms,y[:,j],linestyle='dashed')

fig = figure()
#ax = fig.add_subplot(111)
ax = fig.add_axes([0.05, 0.1, 0.8, 0.8])
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
cutoff = max(amax(W),abs(amin(W)))
im = ax.matshow(W,cmap=mpl.cm.coolwarm,vmin=-cutoff,vmax=cutoff)
fig.colorbar(im,cax=cax)

figure()
#plot(rates.t/second,rates.rate/Hz)
plot(timeseries,poprate,label='popn rate')
plot(timeseries,a0rate,label="rate proj on a0")
legend()
xlabel('time (s)')
ylabel('rate (Hz)')

figure()
for j in range(10):
    plot(Mu.t/second,Mu.u[j]/mV,['r','b'][j/5])
xlabel('time (s)')
ylabel('u (mV)')

show()
