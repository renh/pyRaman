#!/usr/bin/env python

#==============================================================================
# Module documentation
"""
FileName	: pulsed.py
Purpose		: calculate the 1D stimulated Raman spectra of trp-tyr dimer
Author		: Hao Ren (translated from Jason's mathematica notebook)
Version		: 0.1
Date		: Jan 6, 2012
"""
#==============================================================================

#==============================================================================
# Module imports
import numpy as np
from scipy import interpolate
import time
import sys
import os
from common import *
import time
#==============================================================================

def alphaPulsedElement1(j,OmegaL,SYSTEM,PARAM):

    energies = SYSTEM['energies']
    dipoles = SYSTEM['dipoles']
    disps = SYSTEM['disps']
    vibfreqs = SYSTEM['vibfreqs']
    
    basetime = PARAM['basetime']
    num = PARAM['num']
    domega = PARAM['domega']
    beta = PARAM['beta']
    Gamma = PARAM['Gamma']
    sigma = PARAM['sigma']
    vibquanta_max = PARAM['vibquanta_max']

    dt = 2*np.pi/(num * domega)
    omega_list = np.linspace((-num/2+1)*domega,domega*num/2,num)
    t_list = np.linspace(-dt*num/2+dt,dt*num/2,num)
    theta = np.zeros(len(t_list))
    theta[num/2-1:] = 1.0
    #np.save('pth.npy',theta)
    
    nmodes = len(vibfreqs)
    ne = len(energies)
    nt = len(t_list)
    nOmegaL = len(OmegaL)

    sigma = sigma / basetime
    bndwth = np.exp(-(sigma * vibfreqs[j-1]/2)**2)

    MyDisp = np.zeros([ne,nmodes-1])
    MyFreq = np.zeros(nmodes-1)
    
    MyDisp[:,:(j-1)] = disps[:,:(j-1)]
    MyDisp[:,(j-1):] = disps[:,j:]
    MyFreq[:(j-1)] = vibfreqs[:(j-1)]
    MyFreq[(j-1):] = vibfreqs[j:]


    omega_t = np.outer(MyFreq, t_list)

    omega_eg = np.zeros([ne])
    gfunc = np.zeros([ne,nt],dtype="complex_")
    gA = np.zeros([ne,nt])
    gB = np.zeros([ne,nt])

    disp_square = MyDisp * MyDisp * .5
    beta_omega = np.array(
        coth_ufunc(beta * MyFreq * .5),
        dtype='float64'
    )
    cos_omegat = 1.0 - np.cos(omega_t)
    sin_omegat = np.sin(omega_t) - omega_t


    for e in range(ne):
        omega_eg[e] = energies[e] + np.sum(MyDisp[e,:]*MyDisp[e,:] / 2.0 * MyFreq)
        gA[e] = np.dot(
            disp_square[e] * beta_omega,
            cos_omegat
        )
        gB[e] = np.dot(
            disp_square[e],
            sin_omegat
        )

    gfunc = gA + 1.0j * gB
    gfunc = np.exp(
        -gfunc - 1.0j * np.outer(omega_eg, t_list) - Gamma * t_list - t_list*t_list/(4*sigma**2)
    )
    gfunc = dipoles.reshape(-1,1) * gfunc
    tfunc = gfunc * theta
    for e in range(ne):
        dump = np.zeros(nt,dtype="complex_")
        for a in range(7):
            dump += FCfunc(disps[e,j-1]/np.sqrt(2),a,1) * \
                    FCfunc(disps[e,j-1]/np.sqrt(2),a,0) * \
                    np.exp(-1.0j * (a-.5) * vibfreqs[j-1] * t_list)
        tfunc[e] *= dump
    timelist = bndwth * np.add.reduce(tfunc,axis=0)

    alpha = np.zeros(num, dtype="complex_")

    timelist = np.concatenate(
        (timelist[nt/2-1:], timelist[:nt/2-1])
    )
    timelist = np.fft.fft(timelist) / np.sqrt(nt)
    timelist[1:] = timelist[:0:-1]
    alpha = np.concatenate(
        (timelist[nt/2+2:], timelist[:nt/2+2])
    )

    tck_r = interpolate.splrep(omega_list,np.real(alpha))
    tck_i = interpolate.splrep(omega_list,np.imag(alpha))

    alpha_data = np.zeros([nOmegaL],dtype="complex_")
    alpha_data = interpolate.splev(OmegaL,tck_r) + \
            1.0j * interpolate.splev(OmegaL,tck_i)
#    print interpolate.splev(energies[0],tck_r) + \
#            1.0j * interpolate.splev(energies[0],tck_i)

    return alpha_data

# Excitation configuration
# eg.: (i,j) = (1,2) ==> 1st excitation with EX1 and 2nd with EX2 

def ramanPulsed1(OmegaS,conf,alpha,SYSTEM,PARAM):

    energies = SYSTEM['energies']
    ne = len(energies)
    vibfreqs = SYSTEM['vibfreqs']
    nmodes = len(vibfreqs)
    Gamma_g = PARAM['Gamma_g']


#    talpha = np.zeros([ne,nmodes], dtype="complex_")
#    for mode in range(nmodes):
#        talpha[:,mode] = alphaPulsedElement1(mode+1,energies,SYSTEM,PARAM)
#        print "alphaPulsedElement1[%d] done." % (mode+1)
#    #np.save("talpha.npy",talpha)


    talpha1 = alpha[conf[0]-1,:]
    talpha2 = alpha[conf[1]-1,:]

    amplitudes = np.concatenate(
        (talpha2*talpha1, np.conjugate(talpha1)*talpha2)
    )
    #np.save('amp.npy',amplitudes)
    freqs = np.concatenate(
        (vibfreqs, -1.0*vibfreqs)
    )
    spectra = np.zeros(len(OmegaS), dtype="complex_")

    for i in range(len(OmegaS)):
        dump = 0.0j
        for n in range(len(freqs)):
            dump -= (
                ((np.real(amplitudes[n]) * (Gamma_g - 1.0j * OmegaS[i])) + \
                (np.imag(amplitudes[n]) * freqs[n]) ) / \
                (Gamma_g**2 - 2.0j*Gamma_g*OmegaS[i] - OmegaS[i]**2 + freqs[n]**2)
            )
        spectra[i] = dump
    
    spectra /= np.sqrt(2*np.pi)
    return np.abs(spectra)


if __name__ == '__main__':
    PARAM = initialize_param()
    SYSTEM = initialize_system(PARAM)

    energies = SYSTEM['energies']
    ne = len(energies)
    OmegaL = np.linspace(30000,40000,1001)
    nOmegaL = len(OmegaL)
    nmodes = len(SYSTEM['vibfreqs'])
    
    OmegaS_min = PARAM['PU1d_OmegaS_min']
    OmegaS_max = PARAM['PU1d_OmegaS_max']
    nOmegaS = PARAM['PU1d_nOmegaS']
    OmegaS = np.linspace(OmegaS_min, OmegaS_max, nOmegaS)

    read_alpha = False
    if not read_alpha:
        alpha = np.zeros([nOmegaL,nmodes],dtype='complex_')
        print "Calculating alpha elements..."
        for mode in range(nmodes):
            alpha[:,mode] = alphaPulsedElement1(mode+1,OmegaL,SYSTEM,PARAM)
            print "alpha element [%d] " % (mode+1)
        print 'calculate alpha done'
        np.save('talpha.npy',alpha)
    else:
        if not os.path.exists('talpha.npy'):
            print 'alpha data not exist, exit...'
            sys.exit()
        alpha = np.load('talpha.npy')
    
    signals = np.zeros([nOmegaL,len(OmegaS)])
    
    start = time.clock()
    test = ramanPulsed1(OmegaS,(1,201),alpha,SYSTEM,PARAM)
    time_use = time.clock() - start
    print 'Time using for one configuration: ', time_use
    print 'Estimated time using for %d configurations: %g' % (nOmegaL, time_use*nOmegaL)
    for i in range(nOmegaL):
        print "calculating configuration (omega1,omega2) = (%d, %d)" % (OmegaL[100],OmegaL[i])
        signals[i] = ramanPulsed1(OmegaS,(1,i+1),alpha,SYSTEM,PARAM)
#    signals = ramanPulsed1(OmegaS,(1,1),alpha,SYSTEM,PARAM)

    np.save("1DSRR.npy",signals)




