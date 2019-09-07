#!/usr/bin/env python

#==============================================================================
# Module documentation
"""
FileName	: TDRaman.py
Purpose		: calculate the 2D stimulated Raman spectra of trp-tyr dimer
Author		: Hao Ren (translated from Jason's mathematica notebook)
Version		: 0.1
Date		: Jan 6, 2012
#------------------------------------------------------------------------------
Reversion   : 1.0
Date        : Jan 23, 2012
Chglog      : remove all global variables
"""
#==============================================================================

#==============================================================================
# Module imports
import numpy as np
from scipy import interpolate
import time
import sys
from common import *
import os
#==============================================================================

def cal_FCmatrix(SYSTEM,PARAM):
    
    vibquanta_max = PARAM['vibquanta_max']
    energies = SYSTEM['energies']
    vibfreqs = SYSTEM['vibfreqs']
    disps = SYSTEM['disps']
    FCmatrix = np.zeros([len(energies),len(vibfreqs),vibquanta_max+1,vibquanta_max+1])
    for e in range(len(energies)):
        for n in range(len(vibfreqs)):
            for i in range(vibquanta_max+1):
                for j in range(vibquanta_max+1):
                    FCmatrix[e,n,i,j] = FCfunc(disps[e,n]/np.sqrt(2),i,j)
    return FCmatrix


#==========================================================
# alpha element ae1, only one element
#==========================================================
def ae1(OmegaL,SYSTEM,PARAM):
    
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
    #np.save('tl.npy',t_list)
    theta = np.zeros(len(t_list))
    theta[num/2-1:] = 1.0
    sigma = sigma / basetime
    
    ne = len(energies)
    nt = num
    nmodes = len(vibfreqs)
    nOmegaL = len(OmegaL)
    
    omega_eg = np.zeros([ne])
    for e in range(ne):
        omega_eg[e] = energies[e] + np.sum(disps[e,:]*disps[e,:] / 2.0 * vibfreqs)
    
    omega_t = np.outer(vibfreqs, t_list)
    bndwth = 1.

    gfunc = np.zeros([ne,nt],dtype="complex_")
    gA = np.zeros([ne,nt])
    gB = np.zeros([ne,nt])

    disp_square = disps * disps * 0.5
    beta_omega = beta * vibfreqs * 0.5
    beta_omega = np.array(coth_ufunc(beta_omega), dtype="float64")
    cos_omegat = 1.0 - np.cos(omega_t)
    sin_omegat = np.sin(omega_t) - omega_t

    for e in range(ne):
        disp_square_beta_omega = disp_square[e] * beta_omega
        gA[e] = np.dot(
            disp_square_beta_omega,
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
    #np.save('theta.npy',theta)
    tfunc = gfunc * theta
    timelist = bndwth * np.add.reduce(tfunc,axis=0)
    #np.save('ttt.npy',timelist)

    alpha = np.zeros(nt, dtype="complex_")

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


#==========================================================
# alpha element ae2
#==========================================================
def ae2(j,OmegaL,SYSTEM,PARAM):

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
    sigma = sigma / basetime

    ne = len(energies)
    nt = num
    nmodes = len(vibfreqs)
    nOmegaL = len(OmegaL)
    
    MyDisp = np.zeros([ne,nmodes-1])
    MyFreq = np.zeros(nmodes-1)
    MyDisp[:,:(j-1)] = disps[:,:(j-1)]
    MyDisp[:,(j-1):] = disps[:,j:]
    MyFreq[:(j-1)] = vibfreqs[:(j-1)]
    MyFreq[(j-1):] = vibfreqs[j:]

    omega_eg = np.zeros([ne])
    for e in range(ne):
        omega_eg[e] = energies[e] + np.sum(MyDisp[e,:]*MyDisp[e,:] / 2.0 * MyFreq)

    #sys.exit()
    omega_t = np.outer(MyFreq, t_list)
    bndwth = np.exp(-1.0* (sigma * vibfreqs[j-1] * .5)**2)
    #print bndwth

    gfunc = np.zeros([ne,nt],dtype="complex_")
    gA = np.zeros([ne,nt])
    gB = np.zeros([ne,nt])

    disp_square = MyDisp * MyDisp * .5
    beta_omega = beta * MyFreq * .5
    beta_omega = np.array(coth_ufunc(beta_omega), dtype="float64")
    cos_omegat = 1.0 - np.cos(omega_t)
    sin_omegat = np.sin(omega_t) - omega_t

    for e in range(ne):
        disp_square_beta_omega = disp_square[e] * beta_omega
        gA[e] = np.dot(
            disp_square_beta_omega,
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
        for a in range(vibquanta_max+1):
            dump += FCfunc(disps[e,j-1]/np.sqrt(2),a,1) * \
                    FCfunc(disps[e,j-1]/np.sqrt(2),a,0) * \
                    np.exp(-1.0j * (a-.5) * vibfreqs[j-1] * t_list)
        tfunc[e] *= dump

    timelist = bndwth * np.add.reduce(tfunc,axis=0)

    alpha = np.zeros(nt, dtype="complex_")

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


#==========================================================
# alpha element ae5
#==========================================================
def ae5(j,OmegaL,SYSTEM,PARAM):

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
    sigma = sigma / basetime

    ne = len(energies)
    nt = num
    nmodes = len(vibfreqs)
    nOmegaL = len(OmegaL)
    
    MyDisp = np.zeros([ne,nmodes-1])
    MyFreq = np.zeros(nmodes-1)
    MyDisp[:,:(j-1)] = disps[:,:(j-1)]
    MyDisp[:,(j-1):] = disps[:,j:]
    MyFreq[:(j-1)] = vibfreqs[:(j-1)]
    MyFreq[(j-1):] = vibfreqs[j:]

    omega_eg = np.zeros([ne])
    for e in range(ne):
        omega_eg[e] = energies[e] + np.sum(MyDisp[e,:]*MyDisp[e,:] / 2.0 * MyFreq)

    #sys.exit()
    omega_t = np.outer(MyFreq, t_list)
    bndwth = 1.0
    #print bndwth

    gfunc = np.zeros([ne,nt],dtype="complex_")
    gA = np.zeros([ne,nt])
    gB = np.zeros([ne,nt])

    disp_square = MyDisp * MyDisp * .5
    beta_omega = beta * MyFreq * .5
    beta_omega = np.array(coth_ufunc(beta_omega),dtype="float64")
    cos_omegat = 1.0 - np.cos(omega_t)
    sin_omegat = np.sin(omega_t) - omega_t

    for e in range(ne):
        disp_square_beta_omega = disp_square[e] * beta_omega
        gA[e] = np.dot(
            disp_square_beta_omega,
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
        for a in range(vibquanta_max+1):
            dump += FCfunc(disps[e,j-1]/np.sqrt(2),a,1) * \
                    FCfunc(disps[e,j-1]/np.sqrt(2),a,1) * \
                    np.exp(-1.0j * (a-1.0) * vibfreqs[j-1] * t_list)
        tfunc[e] *= dump

    timelist = bndwth * np.add.reduce(tfunc,axis=0)

    alpha = np.zeros(nt, dtype="complex_")

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


#==========================================================
# alpha element ae6
#==========================================================
def ae6(j,k,OmegaL,SYSTEM,PARAM):

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
    sigma = sigma / basetime

    ne = len(energies)
    nt = num
    nmodes = len(vibfreqs)
    nOmegaL = len(OmegaL)
    
    mydisp=[]
    myfreq=[]
    for i in range(nmodes):
        if j == (i+1) or k == (i+1): continue
        mydisp.append(disps[:,i])
        myfreq.append(vibfreqs[i])
    MyDisp = np.transpose(np.array(mydisp))
    MyFreq = np.array(myfreq)

    omega_eg = np.zeros([ne])
    for e in range(ne):
        omega_eg[e] = energies[e] + np.sum(MyDisp[e,:]*MyDisp[e,:] / 2.0 * MyFreq)

    #sys.exit()
    omega_t = np.outer(MyFreq, t_list)
    bndwth = np.exp(-1.0* (sigma * (vibfreqs[j-1] - vibfreqs[k-1]) * .5)**2)

    gfunc = np.zeros([ne,nt],dtype="complex_")
    gA = np.zeros([ne,nt])
    gB = np.zeros([ne,nt])

    disp_square = MyDisp * MyDisp * .5
    beta_omega = beta * MyFreq * .5
    beta_omega = np.array(coth_ufunc(beta_omega),dtype="float64")
    cos_omegat = 1.0 - np.cos(omega_t)
    sin_omegat = np.sin(omega_t) - omega_t

    for e in range(ne):
        disp_square_beta_omega = disp_square[e] * beta_omega
        gA[e] = np.dot(
            disp_square_beta_omega,
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
    omega_jt = vibfreqs[j-1] * t_list
    omega_kt = vibfreqs[k-1] * t_list
    FCmatrix = cal_FCmatrix(SYSTEM,PARAM)
    for e in range(ne):
        dump = np.zeros(nt,dtype="complex_")
        for a in range(vibquanta_max+1):
            for b in range(vibquanta_max+1):
                dump += FCmatrix[e,j-1,a,1] * \
                        FCmatrix[e,j-1,a,0] * \
                        FCmatrix[e,k-1,b,0] * \
                        FCmatrix[e,k-1,b,1] * \
                        np.exp(
                            -1.0j * (a-.5) * omega_jt - \
                             1.0j * (b-.5) * omega_kt
                        )
        tfunc[e] *= dump

    timelist = bndwth * np.add.reduce(tfunc,axis=0)

    alpha = np.zeros(nt, dtype="complex_")

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

def Cal_alpha(OmegaL,SYSTEM,PARAM):
    vibfreqs = SYSTEM['vibfreqs']
    nmodes = len(vibfreqs)
    nstates = nmodes + 1
    nex = len(OmegaL)
    alpha = np.zeros([nstates,nstates,nex],dtype="complex_")
    for n in range(nstates):
        print "alpha row: %d/%d" % (n+1,nstates) 
        for m in range(n,nstates):
            if n==0 and m==0:
                alpha[n,m] = ae1(OmegaL,SYSTEM,PARAM)
            elif n==0 and m!=0:
                alpha[n,m] = ae2(m,OmegaL,SYSTEM,PARAM)
            elif n!=0 and n==m:
                alpha[n,m] = ae5(n,OmegaL,SYSTEM,PARAM)
            else:
                alpha[n,m] = ae6(n,m,OmegaL,SYSTEM,PARAM)
    for i in range(nstates):
        for j in range(i):
            alpha[i,j] = alpha[j,i]
    return alpha

def term1(a1,a2,a3,Omega1,Omega2,SYSTEM,PARAM):
    nw1 = len(Omega1)
    nw2 = len(Omega2)
    
    vibfreqs = SYSTEM['vibfreqs']
    gamma = PARAM['Gamma_g']
    
    nmodes = len(vibfreqs)
    nstates = nmodes + 1
    gsE = np.concatenate(
        (np.array([0]),vibfreqs)
    )
    
    amp = np.zeros([nstates,nstates],dtype="complex_")
    freqs = np.zeros([nstates,nstates,2])
    for c in range(nstates):
        for cp in range(nstates):
            freqs[c,cp,0] = gsE[c] - gsE[0]
            freqs[c,cp,1] = gsE[c] - gsE[cp]
            if c == cp or c == 0: 
                amp[c,cp] = 0
            else:
                amp[c,cp] = np.conjugate(a2[0,cp]) * a3[cp,c] * a1[c,0]
    w1w2 = np.outer(Omega1,Omega2)
    Omega1_square = Omega1 * Omega1
    Omega2_square = Omega2 * Omega2

    part1 = np.zeros([nw1,nw2],dtype="complex_")
    for n in range(len(freqs)):
        for m in range(len(freqs)):
            app = np.imag(amp[n,m])
            ap = np.real(amp[n,m])
            w01 = freqs[n,m,0]
            w02 = freqs[n,m,1]
            termA = ap*gamma*gamma + app*gamma*w01 + app*gamma*w02 - \
                    ap*w01*w02
            termB = (ap*gamma + app*w02) * Omega1.reshape(-1,1)
            termC = (ap*gamma + app*w01) * Omega2
            termD = ap*w1w2
            
            termE = gamma*gamma + w01*w01 - 2.0j*gamma * Omega1 - Omega1_square
            termF = gamma*gamma + w02*w02 - 2.0j*gamma * Omega2 - Omega2_square

            part1 += (termA - 1.0j * (termB+termC) - termD) / \
                    np.outer(termE,termF)
    return part1/(2*np.pi)

def term2(a1,a2,a3,Omega1,Omega2,SYSTEM,PARAM):
    nw1 = len(Omega1)
    nw2 = len(Omega2)
    
    vibfreqs = SYSTEM['vibfreqs']
    gamma = PARAM['Gamma_g']
    
    nmodes = len(vibfreqs)
    nstates = nmodes + 1
    gsE = np.concatenate(
        (np.array([0]),vibfreqs)
    )

    amp = np.zeros([nstates,nstates],dtype="complex_")
    freqs = np.zeros([nstates,nstates,2])
    for c in range(nstates):
        for cp in range(nstates):
            freqs[c,cp,0] = gsE[0] - gsE[c]
            freqs[c,cp,1] = gsE[0] - gsE[cp]
            if 0 == cp or c == 0: 
                amp[c,cp] = 0
            else:
                amp[c,cp] = np.conjugate(a1[0,c]*a2[c,cp]) * a3[cp,0]
    w1w2 = np.outer(Omega1,Omega2)
    Omega1_square = Omega1 * Omega1
    Omega2_square = Omega2 * Omega2

    part2 = np.zeros([nw1,nw2],dtype="complex_")
    for n in range(len(freqs)):
        for m in range(len(freqs)):
            app = np.imag(amp[n,m])
            ap = np.real(amp[n,m])
            w01 = freqs[n,m,0]
            w02 = freqs[n,m,1]
            termA = ap*gamma*gamma + app*gamma*w01 + app*gamma*w02 - \
                    ap*w01*w02
            termB = (ap*gamma + app*w02) * Omega1.reshape(-1,1)
            termC = (ap*gamma + app*w01) * Omega2
            termD = ap*w1w2
            
            termE = gamma*gamma + w01*w01 - 2.0j*gamma * Omega1 - Omega1_square
            termF = gamma*gamma + w02*w02 - 2.0j*gamma * Omega2 - Omega2_square

            part2 += (termA - 1.0j * (termB+termC) - termD) / \
                    np.outer(termE,termF)
    return part2/(2*np.pi)

def term3(a1,a2,a3,Omega1,Omega2,SYSTEM,PARAM):
    nw1 = len(Omega1)
    nw2 = len(Omega2)
    
    vibfreqs = SYSTEM['vibfreqs']
    gamma = PARAM['Gamma_g']
    
    nmodes = len(vibfreqs)
    nstates = nmodes + 1
    gsE = np.concatenate(
        (np.array([0]),vibfreqs)
    )
  
    amp = np.zeros([nstates,nstates],dtype="complex_")
    freqs = np.zeros([nstates,nstates,2])
    for c in range(nstates):
        for cp in range(nstates):
            freqs[c,cp,0] = gsE[0] - gsE[c]
            freqs[c,cp,1] = gsE[cp] - gsE[c]
            if c == cp or c == 0: 
                amp[c,cp] = 0
            else:
                amp[c,cp] = np.conjugate(a1[0,c]) * a3[c,cp] * a2[cp,0]
    w1w2 = np.outer(Omega1,Omega2)
    Omega1_square = Omega1 * Omega1
    Omega2_square = Omega2 * Omega2

    part3 = np.zeros([nw1,nw2],dtype="complex_")
    for n in range(len(freqs)):
        for m in range(len(freqs)):
            app = np.imag(amp[n,m])
            ap = np.real(amp[n,m])
            w01 = freqs[n,m,0]
            w02 = freqs[n,m,1]
            termA = ap*gamma*gamma + app*gamma*w01 + app*gamma*w02 - \
                    ap*w01*w02
            termB = (ap*gamma + app*w02) * Omega1.reshape(-1,1)
            termC = (ap*gamma + app*w01) * Omega2
            termD = ap*w1w2
            
            termE = gamma*gamma + w01*w01 - 2.0j*gamma * Omega1 - Omega1_square
            termF = gamma*gamma + w02*w02 - 2.0j*gamma * Omega2 - Omega2_square

            part3 += (termA - 1.0j * (termB+termC) - termD) / \
                    np.outer(termE,termF)
    return part3/(2*np.pi)

def term4(a1,a2,a3,Omega1,Omega2,SYSTEM,PARAM):
    nw1 = len(Omega1)
    nw2 = len(Omega2)
    
    vibfreqs = SYSTEM['vibfreqs']
    gamma = PARAM['Gamma_g']
    
    nmodes = len(vibfreqs)
    nstates = nmodes + 1
    gsE = np.concatenate(
        (np.array([0]),vibfreqs)
    )
    
    amp = np.zeros([nstates,nstates],dtype="complex_")
    freqs = np.zeros([nstates,nstates,2])
    for c in range(nstates):
        for cp in range(nstates):
            freqs[c,cp,0] = gsE[c] - gsE[0]
            freqs[c,cp,1] = gsE[cp] - gsE[0]
            if 0 == cp or c == 0: 
                amp[c,cp] = 0
            else:
                amp[c,cp] = a3[0,cp] * a2[cp,c] * a1[c,0]
    w1w2 = np.outer(Omega1,Omega2)
    Omega1_square = Omega1 * Omega1
    Omega2_square = Omega2 * Omega2

    part4 = np.zeros([nw1,nw2],dtype="complex_")
    for n in range(len(freqs)):
        for m in range(len(freqs)):
            app = np.imag(amp[n,m])
            ap = np.real(amp[n,m])
            w01 = freqs[n,m,0]
            w02 = freqs[n,m,1]
            termA = ap*gamma*gamma + app*gamma*w01 + app*gamma*w02 - \
                    ap*w01*w02
            termB = (ap*gamma + app*w02) * Omega1.reshape(-1,1)
            termC = (ap*gamma + app*w01) * Omega2
            termD = ap*w1w2
            
            termE = gamma*gamma + w01*w01 - 2.0j*gamma * Omega1 - Omega1_square
            termF = gamma*gamma + w02*w02 - 2.0j*gamma * Omega2 - Omega2_square

            part4 += (termA - 1.0j * (termB+termC) - termD) / \
                    np.outer(termE,termF)
    return part4/(2*np.pi)





if __name__ == '__main__':
    PARAM = initialize_param()
    SYSTEM = initialize_system(PARAM)
    nmodes = len(SYSTEM['vibfreqs'])
    OmegaL = PARAM['PU_pulses_list']
    print 'Number of excitation lasers: ', len(OmegaL)
    print 'Excitation lasers:'
    print OmegaL
    
    TD_read_alpha = PARAM['TD_read_alpha']
    if TD_read_alpha:
        if not os.path.exists('TDalpha.npy'):
            die("File 'TDalpha.npy' not found, exit")
        alpha = np.load('TDalpha.npy')
    else:
        print "Calculating alpha matrix..."
        start = time.clock()
        ae1(OmegaL,SYSTEM,PARAM)
        t1 = time.clock() - start
        print "time alpha(0,0): ", t1

        start = time.clock()
        ae2(1,OmegaL,SYSTEM,PARAM)
        t2 = time.clock() - start
        print "time alpha(0,n): ", t2

        start = time.clock()
        ae5(1,OmegaL,SYSTEM,PARAM)
        t3 = time.clock() - start
        print "time alpha(n,n): ", t3
    
        start = time.clock()
        ae6(1,2,OmegaL,SYSTEM,PARAM)
        t4 = time.clock() - start
        print "time alpha(n,m): ", t4

        print "Estimated time for alpha elements: ", t1 + t2*nmodes + t3*nmodes + t4 * \
                nmodes*(nmodes+1)*.5-nmodes
        
        alpha = Cal_alpha(OmegaL,SYSTEM,PARAM)
        np.save('TDalpha.npy',alpha)
        print "calculating alpha done."
        
    conf = PARAM['TD_configuration']
    p1 = int(conf[0])
    p2 = int(conf[1])
    p3 = int(conf[2])
    a1 = alpha[:,:,p1-1]
    a2 = alpha[:,:,p2-1]
    a3 = alpha[:,:,p3-1]
    a1 = a1 / np.linalg.norm(a1,2)
    a2 = a2 / np.linalg.norm(a2,2)
    a3 = a3 / np.linalg.norm(a3,2)

    Omega1_min = PARAM['TD_Omega1_min']
    Omega1_max = PARAM['TD_Omega1_max']
    nOmega1 = PARAM['TD_nOmega1']
    Omega2_min = PARAM['TD_Omega2_min']
    Omega2_max = PARAM['TD_Omega2_max']
    nOmega2 = PARAM['TD_nOmega2']
    
    Omega1 = np.linspace(Omega1_min,Omega1_max,nOmega1)
    Omega2 = np.linspace(Omega2_min,Omega2_max,nOmega2)

    start = time.clock()
    print "Calculating S(i)..."
    S1 = term1(a1,a2,a3,Omega1,Omega2,SYSTEM,PARAM) * 1.E10
    print "Calculating S(ii)..."
    S2 = term2(a1,a2,a3,Omega1,Omega2,SYSTEM,PARAM) * 1.E10
    print "Calculating S(iii)..."
    S3 = term3(a1,a2,a3,Omega1,Omega2,SYSTEM,PARAM) * 1.E10
    print "Calculating S(iv)..."
    S4 = term4(a1,a2,a3,Omega1,Omega2,SYSTEM,PARAM) * 1.E10
    totalspectrum = S1 + S2 + S3 + S4
    print "time for total spectrum: ", time.clock()-start
    np.save("TD-%s.npy"%conf, totalspectrum)
    #np.save("TD-%s-S1.npy"%conf, S1)
    #np.save("TD-%s-S2.npy"%conf, S2)
    #np.save("TD-%s-S3.npy"%conf, S3)
    #np.save("TD-%s-S4.npy"%conf, S4)





