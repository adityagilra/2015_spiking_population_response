# -*- coding: utf-8 -*-
# Linear response with integral equations
# (c) Mar, 2015 Aditya Gilra, EPFL.

from pylab import *
from scipy.optimize import root,bisect
from scipy.integrate import quad,trapz
from scipy.linalg import expm,solve_lyapunov,eig,schur,orth
import pickle

seedW = 106
seed(seedW)
np.random.seed(seedW)

"""
Generate a matrix with a large eigenvalue,
first without maintaining E-I structure,
 i.e. violating Dale's law.
written by Aditya Gilra (c) May 2015.
"""

M = 10
N = 2*M
p = 0.1             # connection probability
R = 2.0
gamma = 3           # ratio of inh to exc
w0 = R/sqrt(p*(1-p)*(1+gamma**2)*2)
w_ex = w0/sqrt(N)
w_in = -gamma*w0/sqrt(N)
eta = 10.           # learning rate
frac_inh = 0.4      # 40% inh connections
                    # how does this relate to p = 0.1?

WisNormal = False   # random EI (non-normal) matrix Hennequin before SOC
#WisNormal = True    # real random normal matrix

W = zeros(shape=(N,N))
W[uniform(size=(N,N))<0.1] = 1.0
W[:,:M] *= w_ex
W[:,M:] *= w_in
if WisNormal:
    # I do a schur decomposition and take only normal part
    #T,Z = schur(W,output="complex")     # output="real" will give 2x2 blocks
    #                                    # on diagonal for complex eigenvalues
    #                                    # Z and T will be real [for real W?]
    #                                    # W = Z T Z^T (W = Z T Z^H for complex)
    #print "check if Z is unitary? norm( dot(conj(transpose(Z)),Z) - I ) =",\
    #                        norm( dot(conj(transpose(Z)),Z) - eye(N) )
    # Using output="complex" and taking only diagonal doesn't work,
    #  as W can become complex.
    #T = diag(diag(T))                   # 1st diag returns vec of diag entries,
    #                                    # 2nd diag converts to 2D matrix
    # Using output="real" and taking only blocks around the diagonal
    #  posibly won't work either as it is the same as above.
    #print "Normality check for real W before. Frobenius norm of (W^T*W - W*W^T) =",\
    #                        norm( dot(transpose(W),W) - dot(W,transpose(W)) )
    #W = dot(dot(Z,T),transpose(conj(Z)))    # take only normal part

    # If I just construct a real symmetric matrix, it has only real eigenvalues.
    # Construct diagonal and compose using orthogonal matrix
    #  as per http://math.stackexchange.com/questions/1155572/a-real-and-normal-matrix-with-all-eigenvalues-complex-but-some-not-purely-imagin
    scaledR = 1.0
    WD = zeros(shape=(N,N))
    for i in range(M):
        # Real values > 0 on diagonal
        repartval = uniform(0,scaledR) # > 0
        idx = 2*i
        WD[idx,idx] = repartval
        WD[idx+1,idx+1] = repartval
        # imaginary values in blocks around diagonal
        impartval = uniform(-scaledR,scaledR) # allowing them to be zero also
        WD[idx,idx+1] = impartval
        WD[idx+1,idx] = -impartval
    orthM = orth(rand(N,N))
    W = dot(dot(orthM,WD),transpose(orthM))
    print "Sum of imaginary parts of 'normal' W =",sum(imag(W))
    print "Normality check for real W after. Frobenius norm of (W^T*W - W*W^T) =",\
                            norm( dot(transpose(W),W) - dot(W,transpose(W)) )

Tmax = 1.0          # 1 second
epsilon = 1e-100    # to set based on energy integral

class Eigen:
    def __init__(self,W,M):
        self.W = W
        self.N = len(W)
        self.I = eye(self.N)
        self.M = M
        
    def solveQ(self):
        return solve_lyapunov((transpose(self.W)-self.I),-2*self.I)

    def make_eig(self):
        self.Q = self.solveQ()
        print "Is Q real? sum(abs(imag(Q))) =",sum(abs(imag(self.Q)))
        print "Is Q symmetric? sum(abs(Q[i,j]-Q[j,i])) =",\
            sum([abs(self.Q[i,j]-self.Q[j,i]) for i in range(self.N) for j in range(i)])
        self.calc_inputs()
        # sorting is not correct
        #if self.lambdas[0] == conj(self.lambdas[1]):
        #    self.lambdas[0] *= 1.0
        #    self.lambdas[1] *= 1.0  # complex conjugate pairs
        #else:
        #    self.lambdas[0] *= 1.0
        #self.W = dot(dot(self.a0s,diag(self.lambdas)),inv(self.a0s))
        #self.W = real( self.W )     # W should be real, im parts negligible
    
    def calc_inputs(self):
        #v,w = schur(self.W,output="complex")
        #                        # eig may not return unitary w even if W is normal
        #                        # schur is same as eig if W is normal ensuring unitary w
        #v = diag(v)             # convert 2D v to 1D v.

        #v,w = eig(self.W)
        #print dot(dot(w,diag(v)),inv(w))

        v,w = eig(self.Q)
        self.lambdas = v                 # eigenvalues of Q
        self.a0s = w                     # directions for max power of response

        #print dot(dot(self.a0s,diag(self.lambdas)),inv(self.a0s))

        # sort doesn't keep complex conj eigenvalues together
        #sortidxs = argsort(v)       # sorts by real part, then imag part
        #sortidxs = sortidxs[::-1]   # highest real part first
        #self.lambdas = v[sortidxs]  # eigenvalues of W
        #self.a0s = w[:,sortidxs]    # directions for eigen response

    def saveW(self,filename):
        pickle.dump( (self.W,self.lambdas,self.a0s),\
                        open( filename, "wb" ) )    
    
if __name__ == '__main__':
    eigen = Eigen(W,M)
    #print eig(eigen.W)[0]
    eigen.make_eig()
    #print eig(eigen.W)[0]
    eigen.saveW('eigenW'+str(seedW)+'M'+str(M)+'normal'+str(WisNormal)+'.pickle')

    fig = figure()
    #ax = fig.add_subplot(111)
    ax = fig.add_axes([0.05, 0.1, 0.8, 0.8])
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    cutoff = max(amax(eigen.W),abs(amin(eigen.W)))
    im = ax.matshow(eigen.W,cmap=cm.coolwarm,\
                                vmin=-cutoff,vmax=cutoff)
    fig.colorbar(im,cax=cax)
    
    figure()
    scatter(real(eigen.lambdas),imag(eigen.lambdas))
    title('eigenvalues of Q (positive, semi-definite)')
    
    show()
