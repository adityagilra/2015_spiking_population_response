from pylab import *

import sys
sys.path.insert(0, '..')

from brian2 import *
stand_alone = False
if stand_alone: set_device('cpp_standalone')
else: prefs.codegen.target = 'weave'

from neurtheor import IntegralPops as IP

recurrent_conn = False#True
SRM0 = True                     # if False, use SRM neurons
                                # quasi-renewal approx for below SRM neurons
                                #  is ~5% lower at 20Hz A0.

if __name__ == '__main__':
    # neuronal constants
    R = 1.0e8*ohm
    tausyn = 100.0e-3*second
    tau0 = 20.0e-3*second
    noise = 5.0e-3*volt
    uth = 20.0e-3*volt
    
    # network constants
    N = 10000
    connprob = 0.1
    I0 = 0.0e-3*mV/R            # V/Ohm = A
    #totalw_pernrn = 15.0e-3     # V, recurrent weight
    totalw_pernrn = 0.0e-3*volt # V, recurrent weight
                                # if I0 = 10mV/R, and noise = 5 mV,
                                #  then totalw_pernrn = 15mV/R is ~ limit
                                #  before activity blow up at 20mV/R
    w0 = totalw_pernrn/connprob/N
    win = 1.0e-3*volt           # V, input weight for rate below

    # stimulus constants
    rate0 = 100*Hz
    ratemod = 50*Hz
    stimfreq = 5*Hz
    fullt = 1.0*second
    tstep = defaultclock.dt
    print "time step for simulation is",tstep

    ################# BRIAN2 simulation #####################

    # reset eta acts as a threshold increase
    model_eqns = """
        u = (I + K) * R : volt
        I = I0 : amp
        exph = 1.0/tau0*exp((u-eta)/noise) : Hz
        deta/dt = -eta/tau0 : volt
        dK/dt = -K/tausyn : amp
    """
    threshold_eqns = "rand()<=exph*tstep"
    if SRM0: # SRM0 (exact renewal process)
        reset_eqns = "eta=uth"
    else: # usual SRM (approx as quasi-renewal process)
        reset_eqns = "eta+=uth"
    seed(100)
    np.random.seed(100)
    # the hazard function rho is the firing rate,
    #  in time dt the probability to fire is rho*dt.
    # noise below is only the output noise,
    #  input spiking noise comes from spiking during the simulation
    Nrns = NeuronGroup(N, model_eqns, \
                        threshold=threshold_eqns,\
                        reset=reset_eqns)
    #Nrns.I = np.random.uniform((uth-noise)/R/amp,\
    #                           (uth+noise)/R/amp,N)*amp
    #                           # uniform does not retain units, hence explicit
    #Nrns.I = I0

    #NrnsIn = PoissonGroup(N,rates='rate0+ratemod*sin(2*pi*stimfreq*t)')
                                # using ratemod*sin(2*pi*stimfreq*t) doesn't work!
                                #  hence using TimedArray()
    tarray = arange(0,fullt/second,tstep/second)
    ratearray = rate0/Hz + ratemod/Hz*sin(2*pi*stimfreq/Hz*tarray)
    #ratearrayN = repeat([ratearray],N,axis=0) # neuron-number is row, time is col
    ratestim = TimedArray(ratearray*Hz,dt=tstep)
    #NrnsIn = PoissonGroup(N,rates='ratestim(t)')
                                # nopes, somehow it cannot take time dependent values!
                                # tried pure (t) and (t,i), nothing works!
                                # just takes the value at t=0.
    NrnsIn = NeuronGroup(N, 'rho = ratestim(t) : Hz', \
                        threshold='rand()<rho*tstep')
                                # PoissonGroup doesn't seem to work with TimedArray,
                                #  just use NeuronGroup.
    conns = Synapses(NrnsIn,Nrns,pre='K += win/R')
    conns.connect('i==j')

    if recurrent_conn:
        Syns = Synapses(Nrns, Nrns, 'w : amp', pre='K += w')
        Syns.connect(True, p=connprob)
        Syns.w = w0/R

    spikes = SpikeMonitor(Nrns)
    rates = PopulationRateMonitor(Nrns)
    ratesIn = PopulationRateMonitor(NrnsIn)
    run(fullt,report='text')
    if stand_alone:
        device.build(directory='output', compile=True,\
                                        run=True, debug=False)

    #plot(spikes.t/second, spikes.i, '.')
    #A0sim = sum(len(spikes.i[where(spikes.t<settletime)[0]]))/float(N)/settletime
    #print "Average rate per neuron at baseline =", A0sim

    #################### Integral approach and plotting ####################

    intPop = IP.IntegralPop(N,I0*R/volt,tau0/second,\
                uth/volt,0.0,noise/volt,\
                w0/volt,tausyn/second,connprob,win/volt,rate0/Hz)

    dt = intPop.integrate_dt
    mpoints = int(intPop.tupper/dt)
    tarray = arange(0,fullt/second,dt)
    dratearray = ratemod/Hz*sin(2*pi*stimfreq/Hz*tarray)

    intPop.get_background_rate()
    print "The background rate is",intPop.A0
    print "The h is",intPop.h
    print "synint",intPop.kernelsyntildeIntegral
    
    print 'Evolving rate input'
    Avec = intPop.evolve(tarray,dratearray,mpoints)
    
    print 'Convolving linear rate input'
    Aveclin = intPop.lin_response_rate(dratearray,dt)

    figure()
    plot(tarray,intPop.harray,color='blue',label='h (V)')
    ylabel('h (V)',color='blue')
    twinx()
    
    binunits = 10
    bindt = tstep*binunits
    bins = range(int(fullt/bindt))
    Nbins = len(bins)
    plot([rates.t[i*binunits]/second+bindt/2.0/second\
                                for i in bins],\
        [sum(rates.rate[i*binunits:(i+1)*binunits]/Hz)/float(binunits)\
                                for i in bins],\
        ',-g',label='sim')
    plot([ratesIn.t[i*binunits]/second+bindt/2.0/second\
                                for i in bins],\
        [sum(ratesIn.rate[i*binunits:(i+1)*binunits]/Hz)/float(binunits)\
                                for i in bins],\
        ',-c',label='inp')

    plot(tarray,Avec,color='red',label='integ evolve')
    plot(tarray, Aveclin,color='magenta',label='lin evolve')
    ylabel('rate (Hz)',color='red')
    legend()
    
    figure()
    tarray = arange(0.0,intPop.kernelInf,intPop.integrate_dt)
    plot(tarray,intPop.kernelsyntilde(tarray))
    
    show()
