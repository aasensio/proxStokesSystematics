from __future__ import print_function
import numpy as np
import matplotlib.pyplot as pl
import pyiacsun as ps
from ipdb import set_trace as stop
import scipy.linalg as sl
import scipy.special as sp
import scipy.optimize as op
import prox
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

def lowLimitThr(x, low):
    x[x < low] = 0.0
    return x

def softThrHessian(hessian, gradient, xk, lambdaPar, lower):
    """
    Solve the quadratic non-convex sub-problem obtained from the global optimization problem
    f(x) = g(x) + h(x)

    with g(x) a smooth function and h(x) a possibly non-smooth with a simple proximal operator.

    The subproblem to be solved is

    argmin_d grad_g(x_k)^T * d + 1/2 d^T H_k d + h(x_k+d)

    This is solved using FISTA, where the problem is to optimize

    j(d) = k(d) + l(d)

    We use the fact that 

    grad(k(d)) = H_k * d + grad_g(x_k)

    and that l(d) has a simple proximal operator
    
    Args:
        hessian (float): Hessian H_k matrix
        gradient (float): gradient grad_g(x_k) vector
        xk (float): current value of the variable vector
    
    Returns:
        TYPE: the result of the optimization problem
    """

    d = np.zeros_like(xk)
    L = 5e-3
    loop = 0
    t = 1.0
    dOld = np.ones_like(xk)

    while (loop < 50):
        gradK = hessian.dot(d) + gradient
        dnew = d - L * gradK
        dnew[lower:] = softThrShifted(dnew[lower:], xk[lower:], lambdaPar)

        tnew = 0.5*(1+np.sqrt(1+4.0*t**2))
        d = dnew + (t-1.0) / tnew * (dnew - d)

        # print("   - Inner subproblem: l2-residual: {0} -- l1: {1}".format(np.linalg.norm(d - dOld), np.linalg.norm(xk+d, 1)))

        # dOld = np.copy(d)       

        loop += 1
    return d

def softThresholdHessianMetric(xk, hessian, lambdaPar, lower=None, upper=None, lipschitz=None):
    """
    Compute the soft thresholding when using a Hessian in the metric. This solves the following
    proximal optimization
    
    prox_h^H(x) = argmin_y h(y) + 0.5*(y-x)^T H (y-x)
    
    The problem is solved with FISTA, making use of the fact that
    
    argmin_y h(y) + g(y)
    
    with h(y) non-smooth and g(y) smooth:
    
        h(y) = lambda * |y|_1
        g(y) = 0.5*(y-x)^T H (y-x)

    To solve this problem, we make use of the fact that we know how to compute the gradient
    of the smooth part and the proximal algorithm of the non-smooth one

        prox_h = soft_thresholding(lambda)
        grad_g = H*(y-x)
    
    Args:
        xk (float): current value of the variable vector
        hessian (float): Hessian H_k matrix
        lambdaPar (float): thresholding parameter
    
    Returns:
        float: the result of the optimization problem
    
    
    """

    y = np.zeros_like(xk)
    
    loop = 0
    t = 1.0

    if (lipschitz):
        tau = 1.0 / lipschitz
    else:
        x1 = np.random.normal(size=y.shape)
        x2 = np.random.normal(size=y.shape)
        gradf1 = hessian.dot(x1-xk)
        gradf2 = hessian.dot(x2-xk)    
        L = np.linalg.norm(gradf1 - gradf2) / np.linalg.norm(x2 - x1)
        tau = 2.0 / L / 5.0
 
    while (loop < 30):
        gradK = hessian.dot(y-xk)
        # ynew = softThr(y - tau * gradK, lambdaPar * tau, lower=lower, upper=upper)
        ynew = hardThr(y - tau * gradK, lambdaPar * tau, lower=lower, upper=upper)

        tnew = 0.5*(1+np.sqrt(1+4.0*t**2))
        y = ynew + (t-1.0) / tnew * (ynew - y)

        t = np.copy(tnew)

        loop += 1
    
    return y

def profile(x, x0, sigma, a=0.0):
    if (a == 0.0):
        prof = np.exp(-(x - x0)**2 / sigma**2)        
    else:
        prof = ps.radtran.voigt((x-x0) / sigma, a)
    return prof / (np.exp(a**2) * sp.erfc(a))


class semiparLTE(object):
    def __init__(self, lambdaL1=None):
        
        np.random.seed(10)

        lowerMask = [35, 128]
        upperMask = [90, 148]

        lowerMask = [34]
        upperMask = [190]

        maskChi2 = []

        for i in range(len(lowerMask)):
            maskChi2.append(np.arange(upperMask[i] - lowerMask[i]+1) + lowerMask[i])

        self.maskChi2 = np.hstack(maskChi2)

        atmos = np.loadtxt('hsra_64.model', skiprows=2)
        lines = np.loadtxt('lines.dat')
        obs = np.load('profiles/singleProfile.npy')


# Normalize continuum
        x= [7,44,190,216,242,244,286]
        y = obs[1,x]        
        coeff = np.polyfit(x, y, 4)
        cont = np.polyval(coeff, np.arange(len(obs[1,:])))

        obs[1:,:] /= cont[None,:]

        # x = np.arange(288)
        # pl.clf()
        # pl.plot(obs[1,:])
        # pl.plot(x, 1.0-0.53*profile(x, 106.75, 2.0, 0.86))
        # pl.xlim([70,120])
        # stop()

    
        self.wavelength = obs[0,self.maskChi2]
        self.fullWavelength = obs[0,:]
        self.contHSRA = ps.util.contHSRA(np.mean(self.wavelength))
        self.obs = obs[1:,self.maskChi2]
        self.nLambda = self.wavelength.shape[0]
        self.nLambdaTotal = obs[0,:].shape[0]

        self.referenceAtmos = atmos

        ps.radtran.initLTENodes(self.referenceAtmos, lines, self.wavelength)
        
        self.noise = 0.01

        if (lambdaL1 == None):
            self.lambdaL1 = self.noise * 50.0
        else:
            self.lambdaL1 = self.noise * lambdaL1
    
# Define number of nodes and set their ranges
        self.nNodes = [5,1,3,0,0,0]
        self.nNodesTotal = np.sum(self.nNodes)

        lower = [-2000.0, 0.01, -5.0, 0.0, 0.0, 0.0]
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

        widths = [1.0, 2.0]

        damping = [0.8, 0.8]

# Telluric lines one        
        left = 50
        right = 170
        for j in range(len(widths)):
            for i in range(right-left):
                pos = left + i
                self.posBasis.append(pos)
                prof = profile(x, pos, widths[j], damping[j])
                self.basis.append(prof / np.linalg.norm(prof, 2))
                self.nBasis += 1

# Continuum
        # for i in range(3):
        #     poly = ((x - 0.5*self.nLambdaTotal) / (0.5*self.nLambdaTotal))**(i+1)
        #     self.basis.append(poly)
        #     self.nBasis += 1
        
        self.basis = np.vstack(self.basis)

        self.basis = self.basis[:,self.maskChi2]        
        
        self.weights = np.asarray([1.0,0.0,0.0,0.0])

        # self.weights = 1.0 / np.max(np.abs(self.obs), axis=1)
        # self.weights /= np.max(self.weights)

        self.factor = 1.0 / (self.nLambda * self.noise**2)
        
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
        
    def computeSystematics(self, x):        
        systematics = np.zeros((4,self.nLambda))
        systematics[0,:] = np.sum(x[:,None] * self.basis, axis=0)

        dSystematics = np.zeros((self.nBasis,4,self.nLambda))
        dSystematics[:,0,:] = self.basis
        return systematics, dSystematics
    
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

        systematics, dSystematics = self.computeSystematics(xSys)

        stokes += systematics
        residual = (self.obs - stokes)
        chi2 = np.sum(self.weights[:,None] * residual**2 * self.factor)
        chi2NoWeight = np.sum(residual**2 * self.factor)

        dChi2LTE = -2.0 * np.sum(self.weights[None,:,None] * dStokes * residual[None,:,:] * self.factor, axis=(1,2))
        dChi2Sys = -2.0 * np.sum(self.weights[None,:,None] * dSystematics * residual[None,:,:] * self.factor, axis=(1,2))
        
        ddStokes = dStokes[None,:,:,:] * dStokes[:,None,:,:]
        ddChi2LTE = 2.0 * np.sum(self.weights[None,None,:,None] * ddStokes * self.factor, axis=(2,3))
        
        ddSystematics = dSystematics[None,:,:,:] * dSystematics[:,None,:,:]
        ddChi2Sys = 2.0 * np.sum(self.weights[None,None,:,None] * ddSystematics * self.factor, axis=(2,3))

        return chi2, chi2NoWeight, np.hstack([dChi2LTE,dChi2Sys]), sl.block_diag(ddChi2LTE, ddChi2Sys) 

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
        
        sys, _ = self.computeSystematics(xSys)
        stokes += sys
        residual = (stokes - self.obs)
                                
        return np.sum(self.weights[:,None] * residual**2 * self.factor), np.sum(residual**2 * self.factor)

    def printNodes(self, xLTE):
        xPhysical = self.transformedToPhysical(xLTE)
        nodes = self.vector2Nodes(xPhysical)

        variable = ['T', 'vmic', 'vmac', 'B', 'thetaB', 'phiB']

        for i, n in enumerate(nodes):
            if (len(n) != 0):
                print("   {0} : {1}".format(variable[i], n))

    def computeFinalUncertainty(self, x, p):
        chi2, chi2NW, dChi2, ddChi2 = self.computeFunctionAndGradient(x[0:self.nNodesTotal], x[self.nNodesTotal:])
        H = 0.5 * ddChi2 * 4.0 * self.nLambda
        U, w, VT = np.linalg.svd(H[0:self.nNodesTotal,0:self.nNodesTotal], full_matrices=True)

        wmax = np.max(w)
        wInv = 1.0 / w
        wInv[w < 1e-6*wmax] = 0.0
        covariance = VT.T.dot(np.diag(wInv)).dot(U.T)

        nu = 4.0 * self.nLambda - self.nNodesTotal

        deltaChi2 = op.brentq(lambda x: sp.gammaincc(0.5*nu,0.5*x) + p - 1.0, nu-0.1, 2.0*nu)

        error = np.sqrt(np.diag(covariance[0:self.nNodesTotal,0:self.nNodesTotal]) * deltaChi2)

        return error * self.dtransformedToPhysical(x[0:self.nNodesTotal])

    def optimize(self, acceleration=True, plot=False, fileExtension=None):
        """
        This solves the inversion problem by using the FISTA algorithm
        """
        
        x = np.zeros(self.nNodesTotal+self.nBasis)
        x[0:self.nNodesTotal] = self.physicalToTransformed(self.initial)

        chi2 = 1e10
        chi2Old = 1e20
        relchi2 = np.abs((chi2 - chi2Old) / chi2Old)
        xnew = np.copy(x)
        
        loop = 0        
    
        lambdaLM = 1e-3
        chi2Best = 1e10
        chi2Old = 1e10
        nWorstChi2 = 0

        # dChi2Old = 0

        self.chi2 = []
        self.l0 = []
        self.l1 = []

        while ((relchi2 > 1e-5) & (loop < 20) & (nWorstChi2 < 5)):
            chi2, chi2NW, dChi2, ddChi2 = self.computeFunctionAndGradient(x[0:self.nNodesTotal], x[self.nNodesTotal:])
            chi2Old = np.copy(chi2)

            # if (loop > 0):
            #     yk = (dChi2[0:self.nNodesTotal] - dChi2Old[0:self.nNodesTotal])
            #     yk.shape = (9,1)
            #     sk = deltaxnew
            #     sk.shape = (9,1)
            #     Bk = ddChi2[0:self.nNodesTotal,0:self.nNodesTotal]
                
            #     t1 = yk.dot(yk.T) / (yk.T.dot(sk))
            #     t2 = Bk.dot(sk).dot(sk.T).dot(Bk)
            #     t3 = sk.T.dot(Bk).dot(sk)

            #     ddChi2 = Bk + t1 - t2/t3


            # dChi2Old = dChi2
            
            H = 0.5 * ddChi2            
            H += np.diag(lambdaLM * np.diag(H))
            gradF = 0.5 * dChi2

# We make use of the fact that the Hessian matrix is block diagonal

# First deal with the ME part
            U, w, VT = np.linalg.svd(H[0:self.nNodesTotal,0:self.nNodesTotal], full_matrices=True)

            wmax = np.max(w)
            wInv = 1.0 / w
            wInv[w < 1e-6*wmax] = 0.0

# xnew = xold - H^-1 * grad F
            deltaxnew = -VT.T.dot(np.diag(wInv)).dot(U.T).dot(gradF[0:self.nNodesTotal])
            xnew[0:self.nNodesTotal] = x[0:self.nNodesTotal] + deltaxnew

# Now the linear part
            if (self.nBasis > 0):                

# Calculate the inverse Hessian only one time
                if (loop == 0):
                    U, w, VT = np.linalg.svd(H[self.nNodesTotal:,self.nNodesTotal:], full_matrices=True)
                    wInv = 1.0 / w
                    HSysInv = VT.T.dot(np.diag(wInv)).dot(U.T)
                    self.lipschitz = np.max(w)

                # xnew[self.nNodesTotal:] = x[self.nNodesTotal:] - HSysInv.dot(gradF[self.nNodesTotal:])        
# Now apply the proximal operator            
                # xnew[self.nNodesTotal:] = softThresholdHessianMetric(xnew[self.nNodesTotal:], H[self.nNodesTotal:,self.nNodesTotal:], self.lambdaL1, upper=0.0, lipschitz=self.lipschitz)

                xnew[self.nNodesTotal:] = x[self.nNodesTotal:] - 1.0 / self.lipschitz * gradF[self.nNodesTotal:]
# Now apply the proximal operator            
                xnew[self.nNodesTotal:] = hardThr(xnew[self.nNodesTotal:], self.lambdaL1 / self.lipschitz)
              
            chi2, chi2NW = self.meritFunction(xnew[0:self.nNodesTotal], xnew[self.nNodesTotal:])

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
            l1Norm = np.linalg.norm(x[self.nNodesTotal:], 1)
            l0Norm = np.linalg.norm(x[self.nNodesTotal:], 0)
            
            print("Iteration {0} - chi2={1:10.4f} - l1={2} - l0={3} - relchi2={4} - lambda={5}".format(loop, chi2NW, l1Norm, l0Norm, relchi2, lambdaLM))
            self.printNodes(x[0:self.nNodesTotal])

            self.chi2.append(chi2NW)
            self.l0.append(l0Norm)
            self.l1.append(l1Norm)

                        
            loop += 1

        errorBars = self.vector2Nodes(self.computeFinalUncertainty(x, sp.erf(1.0/np.sqrt(2.0))))
        
        xPhysical = self.transformedToPhysical(x[0:self.nNodesTotal])
        nodes = self.vector2Nodes(xPhysical)
        stokes, cont, atmosNew = ps.radtran.synthLTENodes(self.referenceAtmos, nodes)
        stokes /= self.contHSRA
        
        sys, _ = self.computeSystematics(x[self.nNodesTotal:])

        stokes += sys

        # np.savez( "results/synthesis_lte_lambdaL1_{0}.npz".format(fileExtension), self.obs, stokes, sys, errorBars, atmosNew, self.chi2, self.nodePositions, self.basis, x, self.fullWavelength, 
            # self.posBasis, self.l1, self.l0, self.maskChi2, self.lipschitz)


        pl.close('all')
        
        
        f, ax = pl.subplots(nrows=2, ncols=2)
        ax = ax.flatten()
        labelStokes = ['I/Ic','Q/Ic','U/Ic','V/Ic']
        for i in range(4):
            ax[i].plot(self.obs[i,:])
            ax[i].plot(stokes[i,:])
            if (i == 0):
                ax[i].plot(1.0 + sys[i,:])
            else:
                ax[i].plot(sys[i,:])
            # ax[i].set_xlabel('$\Delta \lambda$ [$\AA$]')
            ax[i].set_ylabel(labelStokes[i])
        pl.tight_layout()
        if (plot):
            pl.savefig('/scratch/Dropbox/CONGRESOS/2015/Hinode9/code/systematicsExampleWithFit.png')

        cmap = sn.color_palette()

        f, ax = pl.subplots(nrows=2, ncols=2)
        ax[0,0].plot(self.referenceAtmos[:,0], self.referenceAtmos[:,1], color=cmap[0])
        ax[0,0].plot(atmosNew[:,0], atmosNew[:,1], color=cmap[1])
        ax[0,0].errorbar(atmosNew[self.nodePositions[0],0], atmosNew[self.nodePositions[0],1], yerr=errorBars[0], fmt='none', ecolor=cmap[1], color=cmap[1], capthick=2)
        ax[0,0].set_xlim([np.min(self.referenceAtmos[:,0])-0.5,np.max(self.referenceAtmos[:,0])+0.5])
        
        ax[0,1].plot(self.referenceAtmos[:,0], self.referenceAtmos[:,3], color=cmap[0])
        ax[0,1].plot(atmosNew[:,0], atmosNew[:,3], color=cmap[1])
        ax[0,1].errorbar(atmosNew[self.nodePositions[2],0], atmosNew[self.nodePositions[2],3], yerr=errorBars[2], fmt='none', ecolor=cmap[1], color=cmap[1], capthick=2)
        ax[0,1].set_xlim([np.min(self.referenceAtmos[:,0])-0.5,np.max(self.referenceAtmos[:,0])+0.5])

        ax[1,0].stem(x[self.nNodesTotal:])
        ax[1,0].set_ylabel(r'$\alpha$')      
        ax[1,0].set_xlabel('i')

        pl.tight_layout()

        # print("--------")
        # print("l1 norm of systematics : {0}".format(np.linalg.norm(x[self.nNodesTotal:], 1)))

        return x


lambdaL1 = [5e3] #[1e2,1e3,5e3,1e4]#,3e5,1e6,3e6] #[10, 100, 1000, 5000] #[1, 10, 100, 1000]
for l in lambdaL1:
    out = semiparLTE(lambdaL1=l)
    res = out.optimize(acceleration=True, plot=False, fileExtension=l)