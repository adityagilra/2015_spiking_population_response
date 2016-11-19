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

recurrent_conn = True
do_integral_approach = True
do_differential_approach = False
SRM0 = True      # if False, use SRM neurons
                 # quasi-renewal approx for below SRM neurons is ~5% lower at 20Hz A0.

##################### simulations of SRM / SRM0 neurons
##################### exc only, recurrent connections

from brian2 import *   # importing brian also does:
                        # 'from pylab import *' which imports:
                        # matplot like commands into the namespace, further
                        # also can use np. for numpy and mpl. for matplotlib
#prefs.codegen.target='numpy'
#prefs.codegen.target='weave'
set_device('cpp_standalone')

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
    totalw_pernrn = 5.0*mV/R
        # if I0 = 10mV/R, and noise = 5 mV,
        #  then totalw_pernrn = 15mV/R is ~ limit
        #  before activity blow up at 20mV/R
else:
    totalw_pernrn = 0.0*amp
w0 = totalw_pernrn/connprob/N

# sim constants
settletime = 2*second
stimtime = 2*second
runtime = settletime + stimtime
tstep = defaultclock.dt
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
    deta/dt = -eta/tau0 : volt
    dK/dt = -K/tausyn : amp
"""
threshold_eqns = "rand()<=1.0/tau0*exp((u-eta)/noise)*tstep"
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
Mu = StateMonitor(Nrns, 'u', record=[0,1])
run(runtime,report='text')
device.build(directory='output', compile=True, run=True, debug=False)

fig = figure(figsize=(14,8))
ax = fig.add_subplot(223)
plot(spikes.t/second, spikes.i, ',')
xlim((1.0,1.5))
xlabel('time (s)')
ylabel('nrn idx')

A0sim = sum(len(spikes.i[where(spikes.t<settletime)[0]]))/float(N)/settletime
print "Average rate per neuron at baseline =", A0sim

#fig = figure()
rateax = fig.add_subplot(224)
binunits = 100
bindt = tstep*binunits
bins = range(int(runtime/bindt))
Nbins = len(bins)
rateax.plot([rates.t[i*binunits]/second+bindt/2.0/second for i in bins],\
    [sum(rates.rate[i*binunits:(i+1)*binunits]/Hz)/float(binunits) for i in bins],
    '.-r',label='sim')
rateax.set_ylabel("rate (Hz)")
rateax.set_xlabel("time (s)")

v_ax = twinx()
v_ax.plot(Mu.t/second, Mu.u[0]/mV,'-,k')
v_ax.plot(Mu.t/second, Mu.u[1]/mV,'-,k',label="Vm")
v_ax.set_ylabel('voltage (mV)')

################## Integral equation approach

if do_integral_approach:
    from scipy.integrate import quad,simps
    from scipy.optimize import fsolve

    eps = np.finfo(float).eps
    integrate_dt = 1e-3 # seconds

    # Not easy to follow units outside of Brian.
    # Be very careful about passing unit-ful variables to numpy functions
    
    ######## background / baseline calculation
    noise2 = noise**2
    w02factor = 0.5*connprob*N*(tausyn/second)*(w0*R)**2

    eta_range = arange(0.0,1.0,integrate_dt)
    #fig = figure()
    basicax = fig.add_subplot(221)
    xlabel('time (s)')
    ylabel('see legend')

    def eta(s):
        return uth*exp(-s/tau0SI)      # reset as a threshold increase
    basicax.plot(eta_range,eta(eta_range)/noise,color='b',label='eta (x noise)')

    def gamma_tau(t,tau,noise):
        return -(np.expm1(-eta(t+tau)/noise))*noise
                        # don't multiply by (t>=tau) unlike Moritz,
                        #  since my integration limits are different
                        # expm1(x) = exp(x) - 1, for more accuracy if x is small
    #basicax.plot(eta_range,gamma_tau(eta_range,0.1,noise)/noise,color='r',label='gamma_(tau=0.1s) (x noise)')
    
    def avg_eta(s,A0,noise):
        # average accumulation of previous 'resets'
        # this leads to some adaptation
        #return quad(gamma_tau,0.0,1.0,args=(s,noise))[0] * A0
                                        # quad returns (integral, errorbound)
        return array([sum(gamma_tau(eta_range,t,noise))/volt for t in s])*volt \
                    * integrate_dt * A0 # array() loses units, hence /volt...*volt
    #basicax.plot(eta_range,avg_eta(eta_range,A0sim/Hz,noise)/noise,color='g',label='avg_eta (x noise)')

    def hazard(s,h0,A0):
        ''' 1/time units (1/tau0), converted to 1/second
            h0 must have units of brian volt 
            s should be in seconds (but as not brian units) '''
        noisetot = sqrt(noise2+w02factor*A0)
        if SRM0:
            theta0 = eta(s)
        else: # SRM
            theta0 = eta(s) + avg_eta(s,A0,noisetot)
        return 1/tau0SI*exp((h0-theta0)/noisetot)
        #return 0.5*(sign(s-5e-3)+1) * 200 # 200 Hz * H(s-refract)
                                        # constant hazard with refractory period
    basicax.plot(eta_range,hazard(eta_range,\
                    (I0+totalw_pernrn*tausyn/second*A0sim/Hz)*R,\
                    A0sim/Hz)/Hz/100,\
                    color='c',label='hazard (x100Hz)')

    def survivor(s,h0,A0):
        ''' survivor function (unitless) is a survival probability (S=1 at t=t_hat)
            (it is not a probability density, hence not integral-normalized)
            s should be in seconds (but not as brian units) '''
        #return exp(-quad(hazard,0.0,s,args=(h0,A0))[0])
                                        # quad returns (integral, errorbound)
        return array([exp(-sum(hazard(arange(0.,t,integrate_dt),h0,A0))*integrate_dt) \
                        for t in s])
    basicax.plot(eta_range[:100],survivor(eta_range[:100],\
                    (I0+totalw_pernrn*tausyn/second*A0sim/Hz)*R,\
                    A0sim/Hz)*10,color='m',label='survivor S0 (x10)')

    # upper limit of integration for survivor integral depends on how fast survivor is dropping.
    # check how fast the survivor integral is dropping
    #  and use it to set the upper limit for the survivor integral (see comments in g_sigma()).
    tupper = 0.1
    survivor_neglect = 1e-4
    while tupper<1e100:
        if survivor([tupper],I0*R,0.0)[0] < survivor_neglect: break # use this value of tupper
        tupper *= 2.0
    survivor_integrationsteps = 100
    fullt = linspace(0.,tupper,survivor_integrationsteps)
    fullt_dt = fullt[1]-fullt[0]

    print "Survivor probability drops to <",survivor_neglect,"by tupper =",tupper,"seconds."

    def g_sigma(h0,A0,tupper):
        # SI units second for integration variable
        # quad returns (integral, errorbound), hence take [0]
        # Ideally, I should integrate to infinity `inf`,
        # but for low input, survivor ~= 1, and the integral diverges.
        # If you integrate to 1e4, then for reasonable input that causes spiking,
        #  the constraint optimization doesn't converge.
        # Better instead to see how fast the survivor integral is dropping
        #  and use it to set the upper limit, thus sent in from outside (see below).
        #survivor_integral = quad(survivor,0.0,tupper,args=(h0,A0))[0]
        survivor_integral = sum(survivor(fullt,h0,A0))*fullt_dt
        if survivor_integral > eps: 
            return 1.0/survivor_integral
        else: return 1.0/eps

    def constraint(x,*args):
        A0 = x[0]
        tupper = args[0]
        # A0 assumed in Hz,
        # Need to use synaptic weight*tausyn/1second to obtain avg of exponential synapse
        # see eq (14.8) in Wulfram's book2
        return ( A0 - g_sigma( R * (I0 + totalw_pernrn*tausyn/second*A0), A0, tupper ), )
    
    print "Base rate using sim A0 for noise modifications =", \
        g_sigma( R * (I0 + totalw_pernrn*tausyn/second*A0sim/Hz), A0sim/Hz, tupper ),"Hz."
    # initial value I0*R yields the rate without recurrent connections
    answer = fsolve(constraint,[I0*R,],args=(tupper,),full_output=True)
    A = answer[0]
    print answer[-1]
    
    h0 = R * (I0 + totalw_pernrn*tausyn/second*A[0])
    print "Effective input per neuron =",h0
    print "Effective noise per neuron =",sqrt(noise2+w02factor*A[0])
    print "Base population activity from population evolution consistency is A0 =",A[0],"Hz."

    # we can also use the mean ISI and invert it to get the rate.
    def ISI(s,h0,A0):
        return hazard(s,h0,A0)*survivor(s,h0,A0)
        
    def ISIprime(s,h0,A0):
        return -hazard(s,h0,A0)**2.0 * survivor(s,h0,A0)

    ISIdistrib = ISI(fullt,h0,A0sim/Hz)
    basicax.plot(fullt,ISIdistrib/10.0,color='y',label='ISIdistrib P0 (/10)')
    normP0 = sum(ISIdistrib)*fullt_dt
    print "Norm of ISI distribution =",normP0
    meanISI = sum(fullt*ISI(fullt,h0,A0sim/Hz))*fullt_dt
    print "1/meanISI =",1./meanISI,"Hz."
    basicax.set_xlim([0.0,0.05])
    legend(loc='upper right',numpoints=1,\
        labelspacing=0.0,borderpad=0.01,markerscale=0.5,columnspacing=0.0,\
        handletextpad=0.1,prop={'size':14},frameon=False)

    ######### Linear response

    deltaA = zeros(Nbins)
    # inf approx for time
    tupinf = 2.0 # seconds
    tlowinf = -2.0 # seconds
    bindt = binunits*tstep/second

    # I approximate deltaIarray by linear interpolation
    def L_integrand(tprime,x,h0,A0):
        return hazard(tprime,h0,A0)/sqrt(noise2+w02factor*A0)\
                        *survivor(tprime+x,h0,A0) * volt*second
                                        # convert to SI units

    def L_SRM(x,h0,A0):
        '''
        L_SRM is independent of current time
        See this by substituting t' = s-t_hat in eqn (14.58)
        as also seen from eqn (14.54) in Wulfram's book2
        '''
        #return (x>=0) * quad(L_integrand,0,tupinf,args=(x,h0,A0))[0]
        return (x>=0) * sum(L_integrand(fullt,x,h0,A0))*fullt_dt
            # Heaviside(x) * integral

    # plot the linear response kernel L_SRM(x)
    xlow = 1e-100
    xhigh = 0.1
    xdt = 0.005
    trange = arange(xlow,xhigh,xdt)
    numx = len(trange)
    L_SRMarray = array([L_SRM(tpt,h0,A[0]) for tpt in trange])

    print "Linear response kernel L_SRM(x) computed."
    
    # calculate the linear response for the current perturbation

    def L_SRMinterp(x):
        '''
        returns linearly interpolated values of linear filter for SRM
        from the already calculated array L_SRMarray for given h0.
        '''
        idx = int((x-xlow)/xdt)
        if x<0: return 0.0
        elif idx>=(numx-1): return 0.0 # assume response decays to zero
        else:
            (Llow,Lhigh) = L_SRMarray[idx:idx+2]
            return (Llow + (x/xdt-idx)*(Lhigh-Llow))

    #figure()
    fig.add_subplot(222)
    plottrange = arange(xlow,xhigh,xdt/100.)
    plot(plottrange,[L_SRMinterp(x) for x in plottrange],color='b')
    xlabel('time (s)')
    ylabel('linear SRM response')
    xlim([0.,0.05])
        
    def ISIinterp(x):
        '''
        returns linearly interpolated values of P0
        from the already calculated array ISIdistrib for given h0.
        '''
        idx = int(x/fullt_dt)
        endidx = len(fullt)
        if x<0: return 0.0
        elif idx>=(endidx-1): return 0.0 # assume ISIdistrib decays to zero
        else:
            (P0low,P0high) = ISIdistrib[idx:idx+2]
            return (P0low + (x/fullt_dt-idx)*(P0high-P0low))

    def P0_deltaA(t_hat,bini,h0,A0):
        t = bini*bindt
        return survivor([t-t_hat],h0,A0)[0]*deltaA[bini]

    def constraintdelta(args,bini,h0,dermod):
        deltaA = args[0]
        error = deltaA - quad(P0_deltaA,0.0,bini*bindt,args=(bini,h0,A[0]))[0] \
                - A[0]*dermod # A0 is base activity calculated above
        return error

    def deltaI_interp(t):
        '''
        returns linearly interpolated values of h0
        from the already calculated deltaIarray.
        '''
        tidx = int(t/tstep/second)
        extrafraction = t/(tstep/second)-tidx        
        if tidx<0: deltaI = 0.0
        else:
            (deltaIlow,deltaIhigh) = deltaIarray[tidx:tidx+2]
            deltaI = (deltaIlow + extrafraction*(deltaIhigh-deltaIlow))/amp
        return deltaI

    def L_deltah(x,bini,h0):
        tminusx = bini*binunits*tstep/second - x
        # linearly interpolate deltaI
        deltaI = deltaI_interp(tminux)
        #return L_SRM(x,bini,h0)*R/ohm*deltaI # L(x)*deltah(t-x), removed units
        # no need to re-calculate L_SRM each time
        return L_SRMinterp(x)*R/ohm*deltaI # L(x)*deltah(t-x), removed units

    def mod(bini,h0):
        return quad(L_deltah,0.0,tupinf,args=(bini,h0),epsrel=1e-2)[0] # deltah is zero for t<0

    #lastmod = 0.0
    #for bini in bins:
    #   dermod = (mod(bini,h0)-lastmod)/bindt
    #    answer = fsolve(constraintdelta,[0.0],args=(bini,h0,dermod),full_output=True)
    #    deltaA[bini] = answer[0]
    #    print "t =",bini*bindt,"seconds, deltaA =",answer[0],"Hz."
    #
    #rateax.plot([rates.t[i*binunits]/ms for i in bins],A[0]+deltaA,'.-b',label='linresp')

    deltat = 0.01 # second
    tarray = arange(0.0,runtime/second,deltat)
    deltaAarray = zeros(len(tarray))
    extrapadneeded = len(tarray)-len(ISIdistrib)
    ISIarray = array([ISIinterp(ti) for ti in tarray])
    LSRMarray = array([L_SRMinterp(ti) for ti in tarray])
    deltaharray = R/ohm * array([deltaI_interp(ti) for ti in tarray])
    h0prime = diff(deltaharray) # don't divide by deltat here,
                                #  compensated, as not multiplying by deltat below
    for i,ti in enumerate(tarray[1:],start=1):
        deltaAarray[i] = sum(deltaAarray[:i]*ISIarray[:i][::-1])*deltat + \
                          A[0]*sum(LSRMarray[:i]*h0prime[:i][::-1])

    rateax.plot(tarray,A[0]+deltaAarray,'.-g',label='linevolve')

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
fig.tight_layout()
fig.subplots_adjust(right=0.95)
fig.savefig('Integral_linresponse.png')

show()
