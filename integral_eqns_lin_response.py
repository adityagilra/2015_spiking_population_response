# -*- coding: utf-8 -*-
# Linear response with integral equations

# SRM0 neurons

# simulations

from brian2 import *
prefs.codegen.target = 'numpy'

seed(100)
np.random.seed(100)

R = 1.0e8*ohm
tausyn = 10.0*ms
tau0 = 20.0*ms
N = 1000
runtime = 1.0*second
tstep = defaultclock.dt
Nsteps = runtime/tstep
noise = 10.0*mV
uth = 20.0*mV

h0 = 10.0*mV
I0 = h0/R
deltaIarray = I0/5*array([sin(2*pi*2*Hz*i*tstep) for i in range(Nsteps)])
deltaItimed = TimedArray(deltaIarray,dt=tstep)
model_eqns = """
    I = I0 + deltaItimed(t) : amp
    u = I*R + K : volt
    dK/dt = -K/tausyn : volt
"""
# the hazard function rho is the firing rate, in time dt, the probability to fire is rho*dt.
Nrns = NeuronGroup(N, model_eqns, \
                    threshold="rand()<=1.0/tau0*exp((u-uth)/noise)*tstep",\
                    reset="u-=uth")
#Nrns.I = np.random.uniform((uth-noise)/R/amp,\
#                           (uth+noise)/R/amp,N)*amp # uniform does not retain units, hence explict
#Nrns.I = I0
# I've put in sinusoidal I as part of the equations.

spikes = SpikeMonitor(Nrns)
rates = PopulationRateMonitor(Nrns)
Mu = StateMonitor(Nrns, 'u', record=[0,1])
run(runtime)
plot(spikes.t/second, spikes.i, '.')
print "Average rate per neuron =",sum(spikes.count)/float(N),"Hz."

ratefig = figure()
rateax = ratefig.add_subplot(111)
binunits = 500
bindt = tstep*binunits
bins = range(int(runtime/bindt))
Nbins = len(bins)
rateax.plot([rates.t[i*binunits]/ms+bindt/2.0/ms for i in bins],\
    [sum(rates.rate[i*binunits:(i+1)*binunits]/Hz)/float(binunits) for i in bins],
    '.-r',label='sim')
rateax.set_ylabel("rate (Hz)")
rateax.set_xlabel("time (ms)")

v_ax = twinx()
v_ax.plot(Mu.t/ms, Mu.u[0]/mV,'-,k')
v_ax.plot(Mu.t/ms, Mu.u[1]/mV,'-,k',label="Vm")
v_ax.set_ylabel('voltage (mV)')

# analytical background

from scipy.integrate import quad
from scipy.optimize import fsolve

eps = np.finfo(float).eps

# Not easy to follow units outside of Brian.
# Be very careful about passing unit-ful variables to numpy functions

def hazard(s,h0):
    ''' 1/time units (1/tau0), converted to 1/second
        h0 must have units of brian volt '''
    return 1.0*second/tau0*exp((h0-uth)/noise) # independent of time s here
                                                # (since no refractoriness, no adaptation)
    #return 0.5*(sign(s-5e-3)+1) * 200 # 200 Hz * H(s-refract)
                                                # constant hazard with refractory period

def survivor(s,h0):
    ''' survivor function (unitless) is a survival probability (S=1 at t=t_hat)
        (it is not a probability density, hence not integral-normalized)
        s should be in seconds (but not as brian units) '''
    return exp(-quad(hazard,0.0,s,args=(h0,))[0]) # quad returns (integral, errorbound)

def g_sigma(h0):
    # SI units second for integration variable
    # quad returns (integral, errorbound), hence take [0]
    survivor_integral = quad(survivor,0.0,10.0,args=(h0,))[0]
    if survivor_integral > eps: 
        return 1.0/survivor_integral
    else: return 1.0/eps

def constraint(args):
    A0 = args[0]
    # A0 assumed in Hz,
    # Need to use synaptic weight*tausyn/1second to obtain avg of exponential synapse
    return ( A0 - g_sigma( I0*R ), )
# initial value is the rate without recurrent connections

answer = fsolve(constraint,[I0*R,],full_output=True)
A = answer[0]
print answer[-1]
print "Base population activity is A0 =",A[0],"Hz."
print "The constant hazard function rate is",hazard(0.0,h0),"Hz."
print "The two should be equal!"

# Linear response

deltaA = zeros(Nbins)
# inf approx for time
tupinf = 2.0 # seconds
tlowinf = -2.0 # seconds
bindt = binunits*tstep/second

# I approximate deltaIarray by linear interpolation
def L_integrand(tprime,bini,h0):
    # since L(x) is independent of bini, I comment below calculation, and make it simpler
    #t = bini*bindt
    #return hazard(t-t_hat,h0)/noise*survivor(t-t_hat,h0) * volt*second
    #                                # make it in SI units
    return hazard(tprime,h0)/noise*survivor(tprime,h0) * volt*second
                                    # make it in SI units

def L_SRM(x,bini,h0):
    '''
    L_SRM is independent of current time i.e. bini
    See this by substituting t' = t-t_hat
    as also seen from eqn (14.54)
    '''
    # since L(x) is independent of bini, I comment below calculation, and make it simpler
    #tupper = bini*bindt-x
    #return 0.5*(sign(x)+1) * quad(L_integrand,tlowinf,tupper,args=(bini,h0))[0]
    #    # Heaviside(x) * integral
    return 0.5*(sign(x)+1) * quad(L_integrand,x,tupinf,args=(bini,h0))[0]
        # Heaviside(x) * integral
        # at x=0, sign(x)=0, so Heaviside(x)=1/2

# plot the linear response L_SRM(x)
xlow = 1e-100
xhigh = 1.0
xdt = 0.01
trange = arange(xlow,xhigh,xdt)
numx = len(trange)
L_SRMarray = array([L_SRM(tpt,xhigh,h0) for tpt in trange])
figure()
plot(trange,[survivor(tpt,h0) for tpt in trange],color='r')
ylabel('Survival probability')
twinx()
plot(trange,L_SRMarray,color='b')
ylabel('linear SRM response')

# calculate the linear response for the current perturbation

def L_SRMinterp(x):
    '''
    returns interpolated values of linear filter for SRM
    from the already calculated array L_SRMarray for given h0.
    '''
    idx = int((x-xlow)/xdt)
    if x<0: return 0.0
    elif idx>=(numx-1): return 0.0 # assume response decays to zero
    else:
        (Llow,Lhigh) = L_SRMarray[idx:idx+2]
        return (Llow + (x/xdt-idx)*(Lhigh-Llow))
    
def P0_deltaA(t_hat,bini,h0):
    t = bini*bindt
    return survivor(t-t_hat,h0)*deltaA[bini]

def constraintdelta(args,bini,h0,dermod):
    deltaA = args[0]
    error = deltaA - quad(P0_deltaA,0.0,bini*bindt,args=(bini,h0))[0] \
            - A[0]*dermod # A0 is base activity calculated above
    return error

def L_deltah(x,bini,h0):
    xidx = int(x/tstep/second)
    extrafraction = x/(tstep/second)-xidx
    tminusx_idx = bini*binunits - xidx
    # linearly inerpolate deltaI
    if tminusx_idx<0: deltaI = 0.0
    else:
        (deltaIlow,deltaIhigh) = deltaIarray[tminusx_idx:tminusx_idx+2]
        deltaI = (deltaIlow + extrafraction*(deltaIhigh-deltaIlow))/amp
    #return L_SRM(x,bini,h0)*R/ohm*deltaI # L(x)*deltah(t-x), removed units
    # no need to re-calculate L_SRM each time
    return L_SRMinterp(x)*R/ohm*deltaI # L(x)*deltah(t-x), removed units

def mod(bini,h0):
    return quad(L_deltah,0.0,tupinf,args=(bini,h0),epsrel=1e-2)[0] # deltah is zero for t<0

lastmod = 0.0
for bini in bins:
    dermod = (mod(bini,h0)-lastmod)/bindt
    answer = fsolve(constraintdelta,[0.0],args=(bini,h0,dermod),full_output=True)
    deltaA[bini] = answer[0]
    print "t =",bini*bindt,"seconds, deltaA =",answer[0],"Hz."

rateax.plot([rates.t[i*binunits]/ms for i in bins],A[0]+deltaA,'.-b',label='linresp')

# Compute rate using rate equation (Chapter 15 Wulfram's book)

# Compute rate using LNP (Chapter 15.3.3 Wulfram's book)

rateax.legend()
v_ax.legend()

show()
