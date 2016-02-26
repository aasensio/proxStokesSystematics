from __future__ import print_function
import numpy as np
import matplotlib.pyplot as pl
import pyiacsun as ps
from ipdb import set_trace as stop
import scipy.linalg as sl
import scipy.special as sp
import scipy.optimize as op
import scipy.io as io
import prox
import seaborn as sn
import pywt
import waveletTrans as wl

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

def lowLimitThr(x, low):
    x[x < low] = 0.0
    return x


class inversionWavelet(object):
    def __init__(self, wavelet='wavelet', family='db4', lambdaL1=None, innerIterations=None):

        self.wavelet = wavelet
        if (innerIterations == None):
            self.innerIterations = 100
        else:
            self.innerIterations = innerIterations
        
        np.random.seed(10)

        out = io.readsav('profiles/perfil_andres.sav')
                
        obs = out['perfil']
        
# Fix two bad pixels
        obs[244] = 0.5*(obs[242] + obs[243])
        obs[245] = 0.5*(obs[243] + obs[244])
        obs[246] = 0.5*(obs[244] + obs[245])
                                
        wave = out['lambda']
        
        x = wave[0:630]
        y = obs[0:630]
        xNew = np.linspace(wave[0], wave[630], 512)
        yNew = np.interp(xNew, x, y)

        obs = np.zeros((4,512))
        wave = xNew
        obs[0,:] = yNew

    
        lowerMask = [0]
        upperMask = [511]

        maskChi2 = []

        for i in range(len(lowerMask)):
            maskChi2.append(np.arange(upperMask[i] - lowerMask[i]+1) + lowerMask[i])

        self.maskChi2 = np.hstack(maskChi2)        
    
        self.wavelength = wave[self.maskChi2]
        self.fullWavelength = wave
        self.obs = obs[:,self.maskChi2]
        self.nLambda = self.wavelength.shape[0]
        self.nLambdaTotal = obs[0,:].shape[0]

        self.hazel = ps.radtran.hazel()

        self.noise = 5e-3

        self.lambdaL1 = lambdaL1    

        self.lower = np.asarray([0.0, 3.0, -10.0])
        self.upper = np.asarray([3.0, 15.0, 10.0])
        self.initial = np.asarray([1.0, 7.0, 0.0])
        
        self.weights = np.asarray([1.0,0.0,0.0,0.0])

        self.factor = 1.0 / (self.nLambda * self.noise**2)


        self.synModeInput = 5
        self.nSlabsInput = 1
        self.B1Input = np.asarray([0.0,0.0,0.0])
        self.B2Input = np.asarray([0.0,0.0,0.0])
        self.hInput = 3.e0
        self.tau1Input = 1.e0
        self.tau2Input = 0.e0
        self.boundaryInput  = np.asarray([ps.util.intAllen(10830.0,1.0),0.0,0.0,0.0])
        self.transInput = 1
        self.atomicPolInput = 1
        self.anglesInput = np.asarray([0.0,0.0,90.0])
        self.lambdaAxisInput = self.wavelength - 10829.0911
        self.nLambdaInput = len(self.wavelength)
        self.dopplerWidthInput = 6.e0
        self.dopplerWidth2Input = 0.e0
        self.dampingInput = 0.e0
        self.dopplerVelocityInput = 0.e0
        self.dopplerVelocity2Input = 0.e0
        self.ffInput = 0.e0
        self.betaInput = 1.0
        self.nbarInput = np.asarray([0.0,0.0,0.0,0.0])
        self.omegaInput = np.asarray([0.0,0.0,0.0,0.0])

        self.args = (self.synModeInput, self.nSlabsInput, 
            self.B1Input, self.B2Input, self.hInput, self.tau1Input, self.tau2Input, self.boundaryInput, self.transInput, 
            self.atomicPolInput, self.anglesInput, self.nLambdaInput, self.lambdaAxisInput, self.dopplerWidthInput, 
            self.dopplerWidth2Input, self.dampingInput, self.dopplerVelocityInput, self.dopplerVelocity2Input, 
            self.ffInput, self.betaInput, self.nbarInput, self.omegaInput)

        self.nUnknowns = 3
        self.whichPars = ['tau1', 'vth1', 'v1']

        self.family = family

        if (wavelet == 'wavelet'):            
            self.wavedec, self.waverec = wl.daubechies_factory((self.nLambda), family)
        
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

    def parametersToHazel(self, pars):
        loop = 0

        newPars = list(self.args)

        for p in self.whichPars:            
            if (p == 'B1'):
                newPars[2][0] = pars[loop]
            if (p == 'thB1'):
                newPars[2][1] = pars[loop]
            if (p == 'chiB1'):
                newPars[2][2] = pars[loop]

            if (p == 'B2'):
                newPars[3][0] = pars[loop]
            if (p == 'thB2'):
                newPars[3][1] = pars[loop]
            if (p == 'chiB2'):
                newPars[3][2] = pars[loop]

            if (p == 'tau1'):
                newPars[5] = pars[loop]
            if (p == 'tau2'):
                newPars[6] = pars[loop]

            if (p == 'vth1'):
                newPars[13] = pars[loop]
            if (p == 'vth2'):
                newPars[14] = pars[loop]

            if (p == 'a'):
                newPars[15] = pars[loop]

            if (p == 'v1'):
                newPars[16] = pars[loop]
            if (p == 'v2'):
                newPars[17] = pars[loop]

            if (p == 'ff'):
                newPars[18] = pars[loop]

            if (p == 'beta'):
                newPars[19] = pars[loop]

            loop += 1

        return tuple(newPars)
    
    def computeFunctionAndGradient(self, xHazel, xSys):
        """
        Compute the value of the merit function and of the gradient of the merit function with respect to the
        temperature
        """
        
        xPhysical = self.transformedToPhysical(xHazel)

        args = self.parametersToHazel(xPhysical)

        l, stokes, dStokes = self.hazel.synthDerivatives(self.whichPars, 1e-3, *args)

        sys = np.zeros_like(stokes)
        sys[0,:] = xSys

# Take into account the Jacobian of the transformation
        dStokes *= self.jacobianTransformedParameters(xHazel)[:,None,None]
        
        residual = (self.obs - (stokes + sys))
        chi2 = np.sum(self.weights[:,None] * residual**2 * self.factor)
        chi2NoWeight = np.sum(residual**2 * self.factor)

        dChi2Hazel = -2.0 * np.sum(self.weights[None,:,None] * dStokes * residual[None,:,:] * self.factor, axis=(1,2))
        
        ddStokes = dStokes[None,:,:,:] * dStokes[:,None,:,:]
        ddChi2Hazel = 2.0 * np.sum(self.weights[None,None,:,None] * ddStokes * self.factor, axis=(2,3))
        
        return chi2, chi2NoWeight, dChi2Hazel, ddChi2Hazel, stokes

    def meritFunction(self, xHazel, xSys):
        """
        Compute the value of the merit function for Milne-Eddington parameters and given systematics
        
        Args:
            xHazel (TYPE): Description
            xSys (TYPE): systematics parameters
        
        Deleted Args:
            xMilne (TYPE): Milne-Eddington parameters
        """
        xPhysical = self.transformedToPhysical(xHazel)

        args = self.parametersToHazel(xPhysical)

        l, stokes, _, _ = self.hazel.synth(*args)

        sys = np.zeros_like(stokes)
        sys[0,:] = xSys
                    
        residual = (self.obs - (stokes + sys))
                                
        return np.sum(self.weights[:,None] * residual**2 * self.factor), np.sum(residual**2 * self.factor), stokes

    def printNodes(self, xHazel):
        xPhysical = self.transformedToPhysical(xHazel)
                
        for i, n in enumerate(self.whichPars):
            print("   {0} : {1}".format(n, xPhysical[i]))

    def forwardIUWT(self, x):
        """
        Forward IUWT transform
        """
        dummy = ps.sparse.iuwt_decomposition(x, 6, 0, True)
        return np.vstack((dummy[0],dummy[1][None,:,:]))
    
    def backwardIUWT(self, x):
        """
        Backward IUWT transform
        """
        detail = x[0:-1,:,:]
        smooth = x[-1,:,:]
        return ps.sparse.iuwt_recomposition(detail, 0, smooth)

    def thresholdIUWT(self, x, thr):
        out = np.copy(x)

        detail = x[0:-1,:,:]
        detail[np.abs(detail) < thr] = 0.0
        out[0:-1,:,:] = detail

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
        
        while ((relchi2 > 1e-6) & (loop < 20) & (nWorstChi2 < 5)):

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
            
            chi2, chi2NW, stokes = self.meritFunction(xnew[0:self.nUnknowns], xnew[self.nUnknowns:])
            
            if ((loop + 1) % 3 == 0):
                thr = self.lambdaL1                

                if (self.wavelet == 'iuwt'):
                    tmp = (self.obs[0,:] - stokes[0,:])[:,None]
                    if (self.innerIterations == 1):
                        res = ps.sparse.iuwt_decomposition(tmp, 6, 0, True)                
                        res[0][np.abs(res[0]) < thr] = 0.0                
                        xnew[self.nUnknowns:] = ps.sparse.iuwt_recomposition(res[0], 0, res[1])[:,0]
                    else:                        
                        xnew[self.nUnknowns:] = ps.sparse.proxes.prox_l1General(tmp, self.forwardIUWT, self.backwardIUWT, thr, threshold=self.thresholdIUWT, verbose=False)[:,0]
                if (self.wavelet == 'wavelet'):
                    xnew[self.nUnknowns:] = ps.sparse.proxes.prox_l1General(self.obs[0,:] - stokes[0,:], self.wavedec, self.waverec, thr, threshold='hard', verbose=False)
                
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

            if (self.wavelet == 'iuwt'):
                tmp = self.forwardIUWT(xnew[self.nUnknowns:][:,None])
                l0Norm = np.sum(np.abs(tmp) > 1e-6)
            if (self.wavelet == 'wavelet'):                
                l0Norm = np.sum(np.abs(self.wavedec(x[self.nUnknowns:])) > 1e-10)
            
            print("Iteration {0} - chi2={1:10.4f} - l1={2} - l0={3} - relchi2={4} - lambda={5}".format(loop, chi2NW, l1Norm, l0Norm, relchi2, lambdaLM))
            self.printNodes(x[0:self.nUnknowns])

            self.chi2.append(chi2NW)
            self.l0.append(l0Norm)
            self.l1.append(l1Norm)

                        
            loop += 1

        xPhysical = self.transformedToPhysical(x[0:self.nUnknowns])
        args = self.parametersToHazel(xPhysical)
        l, stokes, _, _ = self.hazel.synth(*args)
                        
        sys = x[self.nUnknowns:]

        np.savez( "results/hazel_{0}_lambda_{1}_inner_{2}.npz".format(self.wavelet,fileExtension,self.innerIterations), self.obs, stokes, sys, self.chi2, x, self.fullWavelength, 
            self.l1, self.l0, self.maskChi2)


        # pl.close('all')

        
        # f, ax = pl.subplots(nrows=2, ncols=2)
        # ax = ax.flatten()
        # labelStokes = ['I/Ic','Q/Ic','U/Ic','V/Ic']
        # ax[0].plot(self.obs[0,:])
        # ax[0].plot(stokes[0,:] + sys)
        # ax[0].plot(1.0+sys)
        # ax[0].plot(stokes[0,:])

        # ax[1].plot(self.obs[0,:] / (1.0+sys))
        # ax[1].plot(stokes[0,:])
        
        # pl.tight_layout()
        # if (plot):
        #     pl.savefig('/scratch/Dropbox/CONGRESOS/2015/Hinode9/code/systematicsExampleWithFit.png')

        
        print("--------")
        print("l1 norm of systematics : {0}".format(np.linalg.norm(x[self.nUnknowns:], 1)))

        return x

lambdas = [3e-3,7e-3,1e-2,3e-2]
for l in lambdas:
    out = inversionWavelet(wavelet='iuwt', lambdaL1=l)
    res = out.optimize(acceleration=True, plot=False, fileExtension=l)

lambdas = [3e-3,7e-3,1e-2,3e-2]
for l in lambdas:
    out = inversionWavelet(wavelet='iuwt', lambdaL1=l, innerIterations=1)
    res = out.optimize(acceleration=True, plot=False, fileExtension=l)

lambdas = [1e-3,1e-2,1e-1,1.0]
for l in lambdas:
    out = inversionWavelet(wavelet='wavelet', family='db8', lambdaL1=l)
    res = out.optimize(acceleration=True, plot=False, fileExtension=l)