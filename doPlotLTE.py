import numpy as np
import matplotlib.pyplot as pl
import seaborn as sn
from ipdb import set_trace as stop
import waveletTrans as wl
import pyiacsun as ps

def plotSingleWavelet(data, row, valueL1):
    obs = data['arr_0']
    stokes = data['arr_1']
    sys = data['arr_2']    
    chi2 = data['arr_3']    
    x = data['arr_4']
    wave = data['arr_5']
    l0 = data['arr_7']

    cmap = sn.color_palette()

    wavedec, waverec = wl.daubechies_factory((len(wave)), 'db8')
    
    ax[row,0].plot(wave, obs[0,:], color=cmap[0])
    ax[row,0].plot(wave, stokes[0,:] + sys, color=cmap[1])
    ax[row,0].plot(wave, 1.0 + sys, '--', color=cmap[2])
    ax[row,0].set_title('$\lambda=${0}'.format(valueL1))
    ax[row,0].set_ylabel('I/I$_c$')
    ax[row,0].set_xlim([10825.8,10833])

    markerline, stemlines, baseline = ax[row,1].stem(wavedec(sys), markerfmt=" ")
    pl.setp(stemlines, 'color', cmap[0])
    pl.setp(baseline, 'color', cmap[0])
    ax[row,1].set_ylabel(r'$\mathbf{W} \alpha$')
    ax[row,1].set_xlim([0,100])
    ax[row,1].set_title('Active={0:4.1f}%'.format(100.0*l0[-1] / len(wave)))

def plotSingleIUWT(data, data2, row, valueL1):
    obs = data['arr_0']
    stokes = data['arr_1']
    sys = data['arr_2']    
    chi2 = data['arr_3']    
    x = data['arr_4']
    wave = data['arr_5']
    l0 = data['arr_7']

    cmap = sn.color_palette()

    ax[row,0].plot(wave, obs[0,:], color=cmap[0])
    ax[row,0].plot(wave, stokes[0,:] + sys, color=cmap[1])
    ax[row,0].plot(wave, 1.0 + sys, '--', color=cmap[2])
    ax[row,0].set_title('$\lambda=${0}'.format(valueL1))
    ax[row,0].set_ylabel('I/I$_c$')
    ax[row,0].set_xlim([6301,6303.2])   

    obs2 = data2['arr_0']
    stokes2 = data2['arr_1']
    sys2 = data2['arr_2']    
    chi22 = data2['arr_3']    
    x2 = data2['arr_4']
    wave2 = data2['arr_5']
    l02 = data2['arr_7']

    ax[row,1].plot(wave2, obs2[0,:], color=cmap[0])
    ax[row,1].plot(wave2, stokes2[0,:] + sys2, color=cmap[1])
    ax[row,1].plot(wave2, 1.0 + sys2, '--', color=cmap[2])
    ax[row,1].set_title('$\lambda=${0}'.format(valueL1))
    ax[row,1].set_ylabel('I/I$_c$')
    ax[row,1].set_xlim([6301,6303.2])

pl.close('all')


# IUWT
lambdaL1 = [0.001,0.005,0.01,0.05]
dat = []
for l in lambdaL1:
    out = np.load("results/lte_iuwt_lambda_{0}_inner_100.npz".format(l))
    dat.append(out)

dat2 = []
for l in lambdaL1:
    out = np.load("results/lte_iuwt_lambda_{0}_inner_1.npz".format(l))
    dat2.append(out)

f, ax = pl.subplots(nrows=len(lambdaL1), ncols=2, figsize=(12,12), sharex='col')
for i in range(len(lambdaL1)):
    plotSingleIUWT(dat[i], dat2[i], i, lambdaL1[i])

ax[-1,0].set_xlabel(r'Wavelength [$\AA$]')
ax[-1,1].set_xlabel(r'Wavelength [$\AA$]')


pl.tight_layout()

pl.savefig('figures/lte_iuwt.pdf')

# db8
# f, ax = pl.subplots(nrows=len(lambdaL1), ncols=2, figsize=(12,12), sharex='col')
# lambdaL1 = [1e-3,1e-2,1e-1,1.0]
# dat = []
# for l in lambdaL1:
#     out = np.load("results/hazel_wavelet_lambda_{0}_inner_100.npz".format(l))
#     dat.append(out)

# for i in range(len(lambdaL1)):
#     plotSingleWavelet(dat[i], i, lambdaL1[i])
# ax[-1,0].set_xlabel(r'Wavelength [$\AA$]')
# ax[-1,1].set_xlabel(r'Wavelength [$\AA$]')

# pl.tight_layout()
# pl.savefig('figures/hazel_wavelet.pdf')
