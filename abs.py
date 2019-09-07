#!/usr/bin/env python

#==============================================================================
# Module documentation
"""
FileName	: cw_Raman.py
Purpose		: calculate the CW Raman spectra of trp-tyr dimer
Author		: Hao Ren (translated from Jason's mathematica notebook)
Version		: 0.1
Date		: Dec 27, 2011
#------------------------------
reversion   : 1.0
Date        : Jan 11, 2012
"""
#==============================================================================

#==============================================================================
# Module imports
import numpy as np
from scipy import interpolate
import time
import sys
from common import *
#==============================================================================


def fullabsorption(SYSTEM,PARAM):
    
    energies = SYSTEM['energies']
    dipoles = SYSTEM['dipoles']
    disps = SYSTEM['disps']
    vibfreqs = SYSTEM['vibfreqs']
    
    basetime = PARAM['basetime']
    num = PARAM['num']
    domega = PARAM['domega']
    beta = PARAM['beta']
    Gamma = PARAM['Gamma']
    vibquanta_max = PARAM['vibquanta_max']
    Gamma = PARAM['Gamma']
    beta = PARAM['beta']
        
    ne = len(energies)
    nt = num
    

    dt = 2*np.pi/(num * domega)
    omega_list = np.linspace((-num/2+1)*domega,domega*num/2,num)
    t_list = np.linspace(-dt*num/2+dt,dt*num/2,num)
    theta = np.zeros(len(t_list))
    theta[num/2:] = 1.0

    # calculate fullabsorption
    # cumulant expansion is used here
    omega_eg = np.zeros([ne])
    for i in range(ne):
        omega_eg[i] = energies[i] + np.sum(disps[i,:]*disps[i,:] / 2.0 * vibfreqs)

    # g(t,e) = gA(t,e) + gB(t,e)
    gfunc = np.zeros([ne,nt], dtype="complex_") 
    gA = np.zeros([ne,nt])
    gB = np.zeros([ne,nt], dtype="complex_")
    
    omega_t = np.outer(vibfreqs,t_list)
    beta_omega = np.array(
        coth_ufunc(beta * vibfreqs * .5),
        dtype = 'float64'
    )
    cos_omegat = 1.0 - np.cos(omega_t)
    sin_omegat = np.sin(omega_t) - omega_t
    disp_square = disps * disps * .5
    
    for e in range(ne):
        gA[e] = np.dot(
            disp_square[e] * beta_omega,
            cos_omegat
        )
        gB[e] = np.dot(
            disp_square[e],
            sin_omegat
        )

    gfunc = gA + 1.0j * gB

    gfunc = np.exp(-gfunc - 1.0j * np.outer(omega_eg, t_list) - Gamma * t_list)
    gfunc = dipoles.reshape(-1,1) * gfunc
    tfunc = gfunc * theta
    
    OmegaL_min = PARAM['ABS_OmegaL_min']
    OmegaL_max = PARAM['ABS_OmegaL_max']
    nOmegaL = PARAM['ABS_nOmegaL']

    sigma = np.zeros([ne,nt], dtype="complex_")
    omega_abs = np.linspace(OmegaL_min,OmegaL_max,nOmegaL)
    sigma_data = np.zeros([ne,nOmegaL])

    for i in range(ne):
        tfunc[i] = np.concatenate( (tfunc[i,nt/2-1:], tfunc[i,:nt/2-1]) )
        tfunc[i] = np.fft.fft(tfunc[i]) / np.sqrt(nt)
        tfunc[i,1:] = tfunc[i,:0:-1]

        sigma[i] = np.concatenate(
            (tfunc[i,nt/2+2:], tfunc[i,:nt/2+2])
        )
        # Only the real part of the Fourier transform contributes to sigma(absorption) 
        tck_r = interpolate.splrep(omega_list,np.real(sigma[i]))
        #tck_i = interpolate.splrep(omega_list,np.imag(sigma[i]))
        sigma_data[i] = interpolate.splev(omega_abs,tck_r) 

    absorption = np.zeros([len(omega_abs),ne+1])
    absorption[:,0] = omega_abs
    absorption[:,1:] = np.transpose(sigma_data)

    return absorption




if __name__ == '__main__':
    PARAM = initialize_param()
    SYSTEM = initialize_system(PARAM)
    
    ne = len(SYSTEM['energies'])
    
    absorption = fullabsorption(SYSTEM,PARAM)

    # write out absorption spectra
    fmt_string = "%12.5f" + "%s" % ("%15.5E" * ne)
    np.savetxt("ABS.dat", absorption, fmt=fmt_string)
    np.save('ABS.npy',absorption)


