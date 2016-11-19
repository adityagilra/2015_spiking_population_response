# -*- coding: utf-8 -*-
# Linear response with integral equations
# (c) Mar, 2015 Aditya Gilra, EPFL.

"""
Unconnected population of identical SRM neurons
 receiving constant + sinusoidal input.
 not SRM0 as the threshold accumulates after each spike
  (for SRM0 the threshold is reset to a fixed value).
Check if the rate response to the sinusoidal fluctuations
 is matched by integral, FP, and other rate approaches.
"""

import sys
sys.path.insert(0, '..')

recurrent_conn = False#True
do_integral_approach = True
do_differential_approach = False
SRM0 = True      # if False, use SRM neurons
                 # quasi-renewal approx for below SRM neurons
                 #  is ~5% lower at 20Hz A0.

##################### simulations of SRM / SRM0 neurons
##################### exc only, recurrent connections

from brian2 import *
prefs.codegen.target = 'numpy'

# neuronal constants
R = 1.0e8*ohm
tausyn = 10.0*ms
tau0 = 20.0*ms
tau0SI = tau0/second
noise = 5.0*mV
uth = 20.0*mV

# network constants
N = 10000
connprob = 0.1
I0 = 10.0*mV/R
if recurrent_conn:
    totalw_pernrn = 15.0*mV/R
        # if I0 = 10mV/R, and noise = 5 mV,
        #  then totalw_pernrn = 15mV/R is ~ limit
        #  before activity blow up at 20mV/R
else:
    totalw_pernrn = 0.0*amp
w0 = totalw_pernrn/connprob/N

# simulation constants
settletime = 1*second
stimtime = 10*second
runtime = settletime + stimtime
tstep = defaultclock.dt
print "time step for simulation is",tstep
Nsteps = int(stimtime/tstep)

deltaIarraysettle = zeros(settletime/tstep)
deltaIarraystim = I0/5/amp * array([sin(2*pi*2*Hz*i*tstep) \
                        for i in range(Nsteps+1)])
deltaIarray = append(deltaIarraysettle,deltaIarraystim) * amp
deltaItimed = TimedArray(deltaIarray,dt=tstep)
# reset eta acts as a threshold increase
model_eqns = """
    u = (I + K) * R : volt
    I = I0 + deltaItimed(t) : amp
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
# I've put in sinusoidal I as part of the equations.

if recurrent_conn:
    Syns = Synapses(Nrns, Nrns, 'w : amp', pre='K += w')
    Syns.connect(True, p=connprob)
    Syns.w = w0

spikes = SpikeMonitor(Nrns)
rates = PopulationRateMonitor(Nrns)
Mu = StateMonitor(Nrns, ('u'), record=range(10))
run(runtime,report='text')
plot(spikes.t/second, spikes.i, '.')
A0sim = sum(len(spikes.i[where(spikes.t<settletime)[0]]))/float(N)/settletime
print "Average rate per neuron at baseline =", A0sim

ratefig = figure()
rateax = ratefig.add_subplot(111)
binunits = 10
bindt = tstep*binunits
bins = range(int(runtime/bindt))
Nbins = len(bins)
rateax.plot([rates.t[i*binunits]/ms+bindt/2.0/ms for i in bins],\
    [sum(rates.rate[i*binunits:(i+1)*binunits]/Hz)/float(binunits) for i in bins],
    ',-r',label='sim')
rateax.set_ylabel("rate (Hz)")
rateax.set_xlabel("time (ms)")

v_ax = twinx()
v_ax.plot(Mu.t/ms, Mu.u[0]/mV,'-,k')
v_ax.plot(Mu.t/ms, Mu.u[1]/mV,'-,k',label="Vm")
v_ax.set_ylabel('voltage (mV)')

################## Integral equation approach

if do_integral_approach:

    from neurtheor.IntegralLinResp import IntegralLinResp

    ######## background / baseline calculation
    noise2 = noise**2
    w02factor = 0.5*connprob*N*(tausyn/second)*(w0*R)**2
    
    ilr = IntegralLinResp(runtime/second,\
                bindt/second,-2.0,2.0,
                N,I0*R/volt,tau0/second,
                uth/volt,0.0,noise/volt,\
                w0*R*N/volt,tausyn/second,connprob,
                0.0,0.0)
    ilr.get_background_rate()
    ilr.compute_linfilter()

    # plot the linear response kernel L_SRM(x)
    figure()
    plot(ilr.trange,ilr.L_SRMarray,color='b')
    xlabel('time (s)')
    ylabel('linear SRM response (V s^2)')
    xlim([0.,0.05])
    print "Linear response kernel L_SRM(x) computed."

    # linear response to stimulus
    deltat = 1e-4
    deltaAarray = ilr.compute_deltaA(deltaIarray*R/volt,tstep/second,deltat)
    rateax.plot(ilr.tarray*1000,ilr.A0+deltaAarray,'.-g',label='linevolve')

    basicfig = figure()
    basicax = basicfig.add_subplot(111)

    basicax.plot(ilr.fullt,ilr.hazard(ilr.fullt,I0*R/volt)/1000,\
                    color='c',label='hazard (kHz)')

    basicax.plot(ilr.fullt,ilr.survivor(ilr.fullt,I0*R/volt)*10,\
                    color='m',label='survivor S0 (x10)')
    
    h0 = R * (I0 + totalw_pernrn*tausyn/second*ilr.A0)
    print "Effective input per neuron =",h0
    print "Effective noise per neuron =",sqrt(noise2+w02factor*ilr.A0)
    print "Base population activity from population evolution"\
                        " consistency is A0 =",ilr.A0,"Hz."

    # we can also use the mean ISI and invert it to get the rate.
    ISIdistrib = ilr.ISI(ilr.fullt,I0*R/volt)
    basicax.plot(ilr.fullt,ISIdistrib/10.0,color='y',label='ISIdistrib P0 (/10)')
    normP0 = sum(ISIdistrib)*ilr.fullt_dt
    print "Norm of ISI distribution =",normP0
    meanISI = sum(ilr.fullt*ilr.ISI(ilr.fullt,I0*R/volt))*ilr.fullt_dt
    print "1/meanISI =",1./meanISI,"Hz."
    #basicax.set_xlim([0.0,0.05])
    legend()

########## Compute rate using rate equation (Chapter 15 Wulfram's book)

########## Compute rate using LNP (Chapter 15.3.3 Wulfram's book)

########## Compute rate using FP approach
# This requires that I have an analogous IF model to the SRM0 model
# I'm keeping that for later. Currently, below code will give errors.
# See ../Ocker_etal_2014_methods/analytic_network_methods.py
# I integrate to find L_SRM as per Richardson 2007 Appendix A

if do_differential_approach:
    # J0, P0: Numerical integration of unperturbed steady state FP equations
    # Using Richardson 2007 Appendix A
    nsteps = 1000
    v0 = theta_reset
    dv = ((theta_reset-v_low))/nsteps
    nreset = int((theta_reset-v_r)/dv)
    def KroneckerDelta(i,j):
        if i==j: return 1.0
        else: return 0.0
    j0 = 1.0
    p0 = 0.0
    v0list = zeros(nsteps)
    j0list = zeros(nsteps)
    p0list = zeros(nsteps)
    Alist = zeros(nsteps)
    Blist = zeros(nsteps)
    for niter in range(nsteps):
        v0 = v0-dv
        v0list[niter] = v0
        Gn = 2.0*(-f(v0)-h0)/sigma**2
        An = exp(dv*Gn)
        Bn = 2.0/sigma**2 * (An-1.0)/Gn
        Alist[niter] = An
        Blist[niter] = Bn
        p0 = p0*An + Bn*tau_m*j0 # before updating j0
        j0 = j0 - KroneckerDelta(niter,nreset)
        j0list[niter] = j0
        p0list[niter] = p0

    p0integral = sum(p0list)*dv
    p0list /= p0integral    # normalize the probability distribution

    figure()
    plot(v0list,p0list,'.-b')
    twinx()
    plot(v0list,j0list,'.-r')
    print "Mean firing rate (slightly off due to discretization) =",1.0/p0integral
    title("P0 (blue), J0 (red)")

    # Numerical Integration of perturbed FP eqn: Full solution, looping over omega
    # From Richardson 2007 Appendix A
    omegalist = array([10**omegaexp for omegaexp in arange(-2,3,0.1)])
    gainlist = zeros(len(omegalist),dtype=complex)
    for oiter,omega in enumerate(omegalist):
        j1free = 1.0
        p1free = 0.0
        v0 = theta_reset
        for niter in range(nsteps):
            v0 = v0-dv
            p1freeSaved = p1free
            p1free = p1free*Alist[niter] + Blist[niter]*tau_m*j1free     # before updating j1free
            j1free = j1free + dv*1j*omega*p1freeSaved - KroneckerDelta(niter,nreset)
        j1driven = 0.0
        p1driven = 0.0
        v0 = theta_reset
        for niter in range(nsteps):
            v0 = v0-dv
            p1drivenSaved = p1driven
            # epsilon set to 1 below, epsilon has voltage units
            p1driven = p1driven*Alist[niter] + tau_m*j1driven + \
                                            R*p0list[niter]*Blist[niter] # before updating j1driven 
            j1driven = j1driven + dv*1j*omega*p1drivenSaved
        gainlist[oiter] = -j1driven/j1free
    freqlist = omegalist/(2*pi)
    figure()
    semilogx(freqlist,absolute(gainlist),'.-b')
    twinx()
    semilogx(freqlist,angle(gainlist),'.-r')
    title("gain ampl(b), phase(r)")

rateax.legend(loc='upper left')
v_ax.legend(loc='lower left')

show()
