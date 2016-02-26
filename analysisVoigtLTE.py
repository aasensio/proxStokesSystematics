from __future__ import print_function
import numpy as np
import matplotlib.pyplot as pl
import pyiacsun as ps
from ipdb import set_trace as stop
import scipy.linalg as sl
import scipy.special as sp
import scipy.optimize as op
import scipy.io as io
import waveletTrans as wl
import seaborn as sn

def softThrShifted(a, b, lambd):
    return np.sign(a) * np.fmax(np.abs(a + b) - lambd, 0) - b

def softThr(x, lambdaPar, lower=None, upper=None):
    out = np.sign(x) * np.fmax(np.abs(x) - lambdaPar, 0)
    if (lower != None):
        out[out < lower] = 0.0
    if (upper != None):
        out[out > upper] = 0.0
    return out

def hardThr(x, lambdaPar, lower=None, upper=None):
    out = np.copy(x)
    out[np.abs(x) < lambdaPar] = 0.0

    if (lower != None):
        out[out < lower] = 0.0
    if (upper != None):
        out[out > upper] = 0.0
    return out

def upLimitThr(x, top):
    x[x > top] = 0.0
    return x

def profile(x, x0, sigma, a=0.0):
    if (a == 0.0):
        prof = np.exp(-(x - x0)**2 / sigma**2)        
    else:
        prof = ps.radtran.voigt((x-x0) / sigma, a)
    return prof / (np.exp(a**2) * sp.erfc(a))



class inversionLTE(object):
    def __init__(self, wavelet='wavelet', family='db4', lambdaL1=None, innerIterations=None):

        self.wavelet = wavelet
        if (innerIterations == None):
            self.innerIterations = 100
        else:
            self.innerIterations = innerIterations
        
        np.random.seed(10)

        obs = np.load('profiles/singleProfile.npy')
# Normalize continuum
        x= [7,44,190,216,242,244,286]
        y = obs[1,x]        
        coeff = np.polyfit(x, y, 4)
        cont = np.polyval(coeff, np.arange(len(obs[1,:])))

        obs[1:,:] /= cont[None,:]

        # obs = obs[:,45:173]
        obs = obs[:,0:256]

        lowerMask = [35]
        upperMask = [192]

        maskChi2 = []

        for i in range(len(lowerMask)):
            maskChi2.append(np.arange(upperMask[i] - lowerMask[i]+1) + lowerMask[i])

        self.maskChi2 = np.hstack(maskChi2)        
    
        self.wavelength = obs[0,self.maskChi2]
        self.fullWavelength = obs[0,:]
        self.contHSRA = ps.util.contHSRA(np.mean(self.wavelength))
        self.obs = obs[1:,self.maskChi2]
        self.nLambda = self.wavelength.shape[0]
        self.nLambdaTotal = obs[0,:].shape[0]

        atmos = np.loadtxt('hsra_64.model', skiprows=2)
        lines = np.loadtxt('lines.dat')

        self.referenceAtmos = atmos

        ps.radtran.initLTENodes(self.referenceAtmos, lines, self.wavelength)
        

        self.noise = 0.01

        self.lambdaL1 = lambdaL1

        # Define number of nodes and set their ranges
        self.nNodes = [5,1,3,0,0,0]
        self.nNodesTotal = np.sum(self.nNodes)
        self.nUnknowns = self.nNodesTotal

        lower = [-2000.0, 0.01, -7.0, 0.0, 0.0, 0.0]
        upper = [2000.0, 5.0, 5.0, 3000.0, 180.0, 180.0]
        initial = [0.0, 1.0, 0.0, 0.01, 20.0, 20.0]

        self.lower = []
        self.upper = []
        self.initial = []

        for i in range(6):
            self.lower.append([lower[i]]*self.nNodes[i])
            self.upper.append([upper[i]]*self.nNodes[i])
            self.initial.append([initial[i]]*self.nNodes[i])


        self.lower = np.hstack(self.lower)
        self.upper = np.hstack(self.upper)
        self.initial = np.hstack(self.initial)
       
        self.nodes = []

        for n in self.nNodes:
            temp = []
            for i in range(n):
                temp.append(0)
            self.nodes.append(temp)
        
        self.nodePositions = ps.radtran.nodePositions(self.referenceAtmos[:,0],self.nodes)

        # Basis set for the systematics
        x = np.arange(self.nLambdaTotal)
        self.nBasis = 0
        self.posBasis = []
        self.basis = [] #np.zeros((self.nBasis,self.nLambda))

        widths = [4.0, 2.0]

        damping = [2.0, 0.0]

# Telluric lines one        
        left = 0
        right = 256
        for j in range(len(widths)):
            for i in range(right-left):
                pos = left + i
                self.posBasis.append(pos)
                prof = profile(x, pos, widths[j], damping[j])
                self.basis.append(prof / np.linalg.norm(prof, 2))
                self.nBasis += 1

        # left = 230
        # right = 380
        # for j in range(len(widths)):
        #     for i in range(right-left):
        #         pos = left + i
        #         self.posBasis.append(pos)
        #         prof = profile(x, pos, widths[j], damping[j])
        #         self.basis.append(prof / np.linalg.norm(prof, 2))
        #         self.nBasis += 1

        # left = 440
        # right = 480
        # for j in range(len(widths)):
        #     for i in range(right-left):
        #         pos = left + i
        #         self.posBasis.append(pos)
        #         prof = profile(x, pos, widths[j], damping[j])
        #         self.basis.append(prof / np.linalg.norm(prof, 2))
        #         self.nBasis += 1

        # self.posBasis.append(150)
        # prof = np.ones_like(x)
        # self.basis.append(prof / np.linalg.norm(prof, 2))
        # self.nBasis += 1

        self.basis = np.vstack(self.basis)

        self.basis = self.basis[:,self.maskChi2].T
        self.basisStar = self.basis.T

        self.tau = 2.0 / np.linalg.norm(self.basis,2)**2
    
        self.weights = np.asarray([1.0,0.0,0.0,0.0])

        self.factor = 1.0 / (self.nLambda * self.noise**2)

        self.family = family

        if (wavelet == 'wavelet'):            
            self.wavedec, self.waverec = wl.daubechies_factory((self.nLambda), family)

        self.nLevelsIUWT = 6
        
    def logit(self, x):
        """
        Logit function
        
        Args:
            x (TYPE): x
        
        Returns:
            TYPE: transformed x
        """
        return np.log(x / (1.0 - x))

    def invLogit(self, x):
        """
        Inverse logit function
        
        Args:
            x (TYPE): x
        
        Returns:
            TYPE: transformed x
        """
        return 1.0 / (1.0 + np.exp(-x))

    def physicalToTransformed(self, x):
        """
        Transform from physical parameters to transformed (unconstrained) ones
        
        Args:
            x (TYPE): vector of parameters
        
        Returns:
            TYPE: transformed vector of parameters
        """
        return self.logit( (x-self.lower) / (self.upper - self.lower))

    def transformedToPhysical(self, x):
        """
        Transform from transformed (unconstrained) parameters to physical ones
        
        Args:
            x (TYPE): vector of transformed parameters
        
        Returns:
            TYPE: vector of parameters
        """
        return self.lower + (self.upper - self.lower) * self.invLogit(x)

    def dtransformedToPhysical(self, x):
        """
        Transform from transformed (unconstrained) parameters to physical ones
        
        Args:
            x (TYPE): vector of transformed parameters
        
        Returns:
            TYPE: vector of parameters
        """
        return (self.upper - self.lower) * np.exp(-x) * self.invLogit(x)**2

    def jacobianTransformedParameters(self, x):
        """
        Compute the Jacobian of the transformation from unconstrained parameters to physical parameters
        
        Args:
            x (TYPE): vector of parameters
        
        Returns:
            TYPE: transformed vector of parameters
        """
        temp = self.invLogit(x)
        return (self.upper - self.lower) * temp * (1.0 - temp)

    def vector2Nodes(self, vector):
        """
        Transform from a vector of parameters to the structure of nodes, made of lists of lists
        
        Args:
            vector (float): model parameters
        
        Returns:
            TYPE: structure of nodes
        """
        nodes = []
        loop = 0

        for n in self.nNodes:
            temp = []
            for i in range(n):
                temp.append(vector[loop])
                loop += 1
            nodes.append(temp)
        return nodes

    def nodes2Vector(self, nodes):
        """Summary
        
        Args:
            nodes (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        return np.asarray([item for sublist in nodes for item in sublist])
    
    def computeFunctionAndGradient(self, xLTE, xSys):
        """
        Compute the value of the merit function and of the gradient of the merit function with respect to the
        temperature
        """
        
        xPhysical = self.transformedToPhysical(xLTE)
        nodes = self.vector2Nodes(xPhysical)
        stokes, cont, atmosNew, dStokes = ps.radtran.synthLTENodes(self.referenceAtmos, nodes, responseFunction=True)

        stokes /= self.contHSRA

        dStokes = self.nodes2Vector(dStokes)
        dStokes /= self.contHSRA

# Take into account the Jacobian of the transformation
        dStokes *= self.jacobianTransformedParameters(xLTE)[:,None,None]

        residual = (self.obs - (stokes + xSys))
        chi2 = np.sum(self.weights[:,None] * residual**2 * self.factor)
        chi2NoWeight = np.sum(residual**2 * self.factor)

        dChi2LTE = -2.0 * np.sum(self.weights[None,:,None] * dStokes * residual[None,:,:] * self.factor, axis=(1,2))
        
        ddStokes = dStokes[None,:,:,:] * dStokes[:,None,:,:]
        ddChi2LTE = 2.0 * np.sum(self.weights[None,None,:,None] * ddStokes * self.factor, axis=(2,3))        

        return chi2, chi2NoWeight, dChi2LTE, ddChi2LTE, stokes


    def meritFunction(self, xLTE, xSys):
        """
        Compute the value of the merit function for Milne-Eddington parameters and given systematics
        
        Args:
            xLTE (TYPE): Description
            xSys (TYPE): systematics parameters
        
        Deleted Args:
            xMilne (TYPE): Milne-Eddington parameters
        """
        xPhysical = self.transformedToPhysical(xLTE)
        nodes = self.vector2Nodes(xPhysical)
        stokes, cont, atmosNew = ps.radtran.synthLTENodes(self.referenceAtmos, nodes)
        stokes /= self.contHSRA

        sys = np.zeros_like(stokes)
        sys[0,:] = xSys

        residual = (self.obs - (stokes + sys))
                                
        return np.sum(self.weights[:,None] * residual**2 * self.factor), np.sum(residual**2 * self.factor), stokes

    def printNodes(self, xLTE):
        xPhysical = self.transformedToPhysical(xLTE)
        nodes = self.vector2Nodes(xPhysical)

        variable = ['T', 'vmic', 'vmac', 'B', 'thetaB', 'phiB']

        for i, n in enumerate(nodes):
            if (len(n) != 0):
                print("   {0} : {1}".format(variable[i], n))

    def forwardGauss(self, x):
        """
        Forward transform induced by the Voigt functions
        """
        return self.basisStar.dot(x)
    
    def backwardGauss(self, x):
        """
        Backward transform induced by the Voigt functions
        """
        return self.basis.dot(x)

    def thresholdGauss(self, x, thr):
        out = np.copy(x)

        out = softThr(out, thr)
        
        # out2 = out[0:-2]
        # out2[out2 > 0.0] = 0.0
        # out[0:-2] = out2

        return out

    def optimize(self, acceleration=True, plot=False, fileExtension=None):
        """
        This solves the inversion problem by using the FISTA algorithm
        """
        
        x = np.zeros(self.nUnknowns+self.nLambda)
        x[0:self.nUnknowns] = self.physicalToTransformed(self.initial)

        chi2 = 1e10
        chi2Old = 1e20
        relchi2 = np.abs((chi2 - chi2Old) / chi2Old)
        xnew = np.copy(x)
        
        loop = 0        
        loopInt = 0
    
        lambdaLM = 1e-3
        chi2Best = 1e10
        chi2Old = 1e10
        nWorstChi2 = 0

        # dChi2Old = 0

        self.chi2 = []
        self.l0 = []
        self.l1 = []
        
        while ((relchi2 > 1e-6) & (loop < 20) & (nWorstChi2 < 8)):

            chi2, chi2NW, dChi2, ddChi2, stokes = self.computeFunctionAndGradient(x[0:self.nUnknowns], x[self.nUnknowns:])

            chi2Old = np.copy(chi2)        
            
            H = 0.5 * ddChi2            
            H += np.diag(lambdaLM * np.diag(H))
            gradF = 0.5 * dChi2

# First deal with the Hazel part
            U, w, VT = np.linalg.svd(H[0:self.nUnknowns,0:self.nUnknowns], full_matrices=True)

            wmax = np.max(w)
            wInv = 1.0 / w
            wInv[w < 1e-6*wmax] = 0.0

# xnew = xold - H^-1 * grad F
            deltaxnew = -VT.T.dot(np.diag(wInv)).dot(U.T).dot(gradF[0:self.nUnknowns])
            xnew[0:self.nUnknowns] = x[0:self.nUnknowns] + deltaxnew
                        
            if ((loop + 1) % 3 == 0):
                thr = self.lambdaL1 
                tmp = self.obs[0,:] - stokes[0,:]                
                xnew[self.nUnknowns:] = ps.sparse.proxes.prox_l1General(tmp, self.forwardGauss, self.backwardGauss, thr, mu=0.9*self.tau, threshold=self.thresholdGauss, verbose=False)
                xnew[self.nUnknowns:] = upLimitThr(xnew[self.nUnknowns:], 0.0)

                #x = np.copy(xnew)

            chi2, chi2NW, stokes = self.meritFunction(xnew[0:self.nUnknowns], xnew[self.nUnknowns:])
                
            if (chi2NW < chi2Best):
                if (lambdaLM >= 1e4):
                    lambdaLM /= 100.0
                elif ((lambdaLM >= 1e-4) or (lambdaLM < 1e4)):
                    lambdaLM /= 10.0
                elif(lambdaLM < 1e-4):
                    lambdaLM /= 5.0
                if (lambdaLM < 1e-6):
                    lambdaLM = 1e-6

                chi2Best = np.copy(chi2NW)
                x = np.copy(xnew)
                nWorstChi2 = 0
            else:
                if (lambdaLM > 1e4):
                    lambdaLM *= 100.0
                elif ((lambdaLM >= 1e-4) or (lambdaLM < 1e4)):
                    lambdaLM *= 10.0
                elif(lambdaLM < 1e-4):
                    lambdaLM *= 5.0
                nWorstChi2 += 1

            relchi2 = np.abs((chi2 - chi2Old) / chi2Old)
            l1Norm = np.linalg.norm(x[self.nUnknowns:], 1)

            tmp = self.forwardGauss(xnew[self.nUnknowns:])
            l0Norm = np.sum(np.abs(tmp) > 1e-6)
            
            print("Iteration {0} - chi2={1:10.4f} - l1={2} - l0={3} - relchi2={4} - lambda={5}".format(loop, chi2NW, l1Norm, l0Norm, relchi2, lambdaLM))
            self.printNodes(x[0:self.nUnknowns])

            self.chi2.append(chi2NW)
            self.l0.append(l0Norm)
            self.l1.append(l1Norm)

                        
            loop += 1

        xPhysical = self.transformedToPhysical(x[0:self.nNodesTotal])
        nodes = self.vector2Nodes(xPhysical)
        stokes, cont, atmosNew = ps.radtran.synthLTENodes(self.referenceAtmos, nodes)
        stokes /= self.contHSRA

        sys = x[self.nUnknowns:]

        np.savez( "results/lte_voigt_lambda_{1}.npz".format(fileExtension), self.obs, stokes, sys, self.chi2, x, self.wavelength, 
             self.l1, self.l0, self.maskChi2)


        # pl.close('all')
        
        # f, ax = pl.subplots(nrows=2, ncols=2, figsize=(12,9))
        # ax = ax.flatten()
        # labelStokes = ['I/Ic','Q/Ic','U/Ic','V/Ic']
        # ax[0].plot(self.obs[0,:])
        # # ax[0].plot(stokes[0,:])
        # ax[0].plot(stokes[0,:] + sys)
        # ax[0].plot(1.0 + sys)

        # cmap = sn.color_palette()

        # markerline, stemlines, baseline = ax[1].stem(self.posBasis, tmp, markerfmt=" ")
        # pl.setp(stemlines, 'color', cmap[0])
        # pl.setp(baseline, 'color', cmap[0])
        
        # pl.tight_layout()

        if (plot):
            pl.savefig('/scratch/Dropbox/CONGRESOS/2015/Hinode9/code/systematicsExampleWithFit.png')

        
        print("--------")
        print("l1 norm of systematics : {0}".format(np.linalg.norm(x[self.nUnknowns:], 1)))

        return x

lambdas = [0.001, 0.005, 0.01, 0.05]
for l in lambdas:
    out = inversionLTE(lambdaL1=l)
    res = out.optimize(acceleration=True, plot=False, fileExtension=l)

# lambdas = [3e-3,7e-3,1e-2,3e-2]
# for l in lambdas:
#     out = inversionWavelet(wavelet='iuwt', lambdaL1=l, innerIterations=1)
#     res = out.optimize(acceleration=True, plot=False, fileExtension=l)

# lambdas = [1e-3,1e-2,1e-1,1.0]
# for l in lambdas:
#     out = inversionWavelet(wavelet='wavelet', family='db8', lambdaL1=l)
#     res = out.optimize(acceleration=True, plot=False, fileExtension=l)