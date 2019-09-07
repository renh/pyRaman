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
Date        : Jan 10, 2012
#------------------------------
reversion   : 1.1
Data        : Jan 23, 2012
Description : move paramenter reading as an external module
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

def alphaCWelement1(j,OmegaL,SYSTEM,PARAM):
    
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

    dt = 2*np.pi/(num * domega)
    omega_list = np.linspace((-num/2+1)*domega,domega*num/2,num)
    t_list = np.linspace(-dt*num/2+dt,dt*num/2,num)
    theta = np.zeros(len(t_list))
    theta[num/2:] = 1.0
    
    nmodes = len(vibfreqs)
    ne = len(energies)
    nt = len(t_list)
    nOmegaL = len(OmegaL)

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
        dtype="float64"
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

    gfunc = np.exp(-gfunc - 1.0j * np.outer(omega_eg, t_list) - Gamma * t_list)
    gfunc = dipoles.reshape(-1,1) * gfunc
    tfunc = gfunc * theta
    for e in range(ne):
        dump = np.zeros(nt,dtype="complex_")
        for a in range(vibquanta_max+1):
            dump += FCfunc(disps[e,j-1]/np.sqrt(2),a,1) * \
                    FCfunc(disps[e,j-1]/np.sqrt(2),a,0) * \
                    np.exp(-1.0j * a * vibfreqs[j-1] * t_list)
        tfunc[e] *= dump
    timelist = np.add.reduce(tfunc,axis=0)

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

    return alpha_data

def cal_alphaCW1(OmegaL,SYSTEM,PARAM):
    print "Calculating alpha elements..."
    nOmegaL = len(OmegaL)
    vibfreqs = SYSTEM['vibfreqs']
    nmodes = len(vibfreqs)
    alphaCW1 = np.zeros([nOmegaL,nmodes],dtype="complex_")
    for j in range(nmodes):
        alphaCW1[:,j] = alphaCWelement1(j+1,OmegaL,SYSTEM,PARAM)
    return alphaCW1
    
    

def ramanCW(iex,OmegaS,alphaCW1,SYSTEM,PARAM):
    vibfreqs = SYSTEM['vibfreqs']
    nmodes = len(vibfreqs)
    Gamma_g = PARAM['Gamma_g']
    
    cw = np.zeros(len(OmegaS))
    for i in range(len(OmegaS)):
        dump = 0.0
        for n in range(nmodes):
            dump += np.abs(alphaCW1[iex,n])**2 * Gamma_g / \
                ((OmegaS[i]-vibfreqs[n])**2 + Gamma_g**2)
        cw[i] = dump
    return cw



if __name__ == '__main__':

    PARAM = initialize_param()
    SYSTEM = initialize_system(PARAM)
    
    energies = SYSTEM['energies']
    vibfreqs = SYSTEM['vibfreqs']
    nmodes = len(SYSTEM['vibfreqs'])
    print 'number of excited states: ', len(energies) 
    print 'number of vibmodes: ', len(vibfreqs)
    
    cal_2DCW = PARAM['CW_cal_2DCW']
    if cal_2DCW:
        # excitation frequency range for 2DCW Raman calculation
        nOmegaL2D = PARAM['CW_nOmegaL']
        OmegaL_min = PARAM['CW_OmegaL_min']
        OmegaL_max = PARAM['CW_OmegaL_max']
        OmegaL2D = np.linspace(OmegaL_min,OmegaL_max,nOmegaL2D)
        # excitation frequencies to which alpha should be calculated
        OmegaL = np.concatenate( 
            (OmegaL2D,energies)
        )
    else:
        #OmegaL = [PARAM['CW_OmegaL']]
        OmegaL = np.arange(30000,44001,100)

    nOmegaL = len(OmegaL)

    read_alpha = PARAM['CW_read_alpha']
    if read_alpha:
        alphaCW1 = np.load('CW-alpha1.npy')
    else:
        alphaCW1 = cal_alphaCW1(OmegaL,SYSTEM,PARAM)
        print "alpha calculation done!"
 
    omega_min = PARAM['CW_OmegaS_min']
    omega_max = PARAM['CW_OmegaS_max']
    nOmegaS = PARAM['CW_nOmegaS']
    
    OmegaS = np.linspace(omega_min,omega_max,nOmegaS)

    cw = np.zeros([nOmegaL,nOmegaS])
    for iex in range(nOmegaL):
        cw[iex] = ramanCW(iex,OmegaS,alphaCW1,SYSTEM,PARAM)
    
    if cal_2DCW:
        TDCW = cw[:nOmegaL2D,:]
        np.save("CW2D.npy",TDCW)

    
    RRdata = np.zeros([nOmegaS,len(OmegaL)+1])
    RRdata[:,0] = OmegaS
    RRdata[:,1:] = np.transpose(cw[-len(OmegaL):,:])
    fmt_string = "%12.5f\t" + "%s" % ("%18.8E "*len(OmegaL))
    #np.savetxt("CWRR.dat",RRdata,fmt=fmt_string)
    np.save("CW2D.npy",RRdata)
        
    
    







