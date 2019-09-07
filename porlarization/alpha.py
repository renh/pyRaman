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
    dipvecs = SYSTEM['dipvecs']
    
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
    
    #gfunc = dipoles.reshape(-1,1) * gfunc
    #print dipoles.reshape(-1,1)
    dip_matrix = np.zeros([3,3,ne])
    for e in range(ne):
        dip_matrix[:,:,e] = np.outer(dipvecs[e],dipvecs[e])
        #print dip_matrix[e]
    gfunc_tensors = np.zeros([3,3,ne,nt],dtype='complex_')
    for x in range(3):
        for y in range(3):
            for e in range(ne):
                gfunc_tensors[x,y,e] = dip_matrix[x,y,e] * gfunc[e]


    #np.save('theta.npy',theta)
    tfunc = np.zeros([3,3,ne,nt],dtype='complex_')
    for x in range(3):
        for y in range(3):
            for e in range(ne):
                tfunc[x,y,e] = gfunc_tensors[x,y,e] * theta
    #tfunc = gfunc * theta

    timelist = np.zeros([3,3,nt])
    timelist = bndwth * np.add.reduce(tfunc,axis=2)

    #timelist = bndwth * np.add.reduce(tfunc,axis=0)
    #np.save('ttt.npy',timelist)

    alpha = np.zeros([3,3,nt], dtype="complex_")
    alpha_data = np.zeros([3,3,nOmegaL],dtype="complex_")
    for x in range(3):
        for y in range(3):
            timelist[x,y] = np.concatenate(
                (timelist[x,y,nt/2-1:], timelist[x,y,:nt/2-1])
            )
            timelist[x,y] = np.fft.fft(timelist[x,y]) / np.sqrt(nt)
            timelist[x,y,1:] = timelist[x,y,:0:-1]
            alpha[x,y] = np.concatenate(
                (timelist[x,y,nt/2+2:], timelist[x,y,:nt/2+2])
            )

            tck_r = interpolate.splrep(omega_list,np.real(alpha[x,y]))
            tck_i = interpolate.splrep(omega_list,np.imag(alpha[x,y]))

            alpha_data[x,y] = interpolate.splev(OmegaL,tck_r) + \
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
    dipvecs = SYSTEM['dipvecs']
    
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

    #gfunc = dipoles.reshape(-1,1) * gfunc
    #tfunc = gfunc * theta
    dip_matrix = np.zeros([3,3,ne])
    for e in range(ne):
        dip_matrix[:,:,e] = np.outer(dipvecs[e],dipvecs[e])
    gfunc_tensors = np.zeros([3,3,ne,nt],dtype='complex_')
    tfunc = np.zeros([3,3,ne,nt],dtype='complex_')
    for x in range(3):
        for y in range(3):
            for e in range(ne):
                gfunc_tensors[x,y,e] = dip_matrix[x,y,e] * gfunc[e]
                tfunc[x,y,e] = gfunc_tensors[x,y,e] * theta
    for e in range(ne):
        dump = np.zeros(nt,dtype="complex_")
        for a in range(vibquanta_max+1):
            dump += FCfunc(disps[e,j-1]/np.sqrt(2),a,1) * \
                    FCfunc(disps[e,j-1]/np.sqrt(2),a,0) * \
                    np.exp(-1.0j * (a-.5) * vibfreqs[j-1] * t_list)
        tfunc[:,:,e] *= dump

    timelist = bndwth * np.add.reduce(tfunc,axis=2)

    alpha = np.zeros([3,3,nt], dtype="complex_")
    alpha_data = np.zeros([3,3,nOmegaL],dtype="complex_")
    for x in range(3):
        for y in range(3):
            timelist[x,y] = np.concatenate(
                (timelist[x,y,nt/2-1:], timelist[x,y,:nt/2-1])
            )
            timelist[x,y] = np.fft.fft(timelist[x,y]) / np.sqrt(nt)
            timelist[x,y,1:] = timelist[x,y,:0:-1]
            alpha[x,y] = np.concatenate(
                (timelist[x,y,nt/2+2:], timelist[x,y,:nt/2+2])
            )

            tck_r = interpolate.splrep(omega_list,np.real(alpha[x,y]))
            tck_i = interpolate.splrep(omega_list,np.imag(alpha[x,y]))

            alpha_data[x,y] = interpolate.splev(OmegaL,tck_r) + \
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
    dip_matrix = np.zeros([3,3,ne])
    for e in range(ne):
        dip_matrix[:,:,e] = np.outer(dipvecs[e],dipvecs[e])
    #gfunc = dipoles.reshape(-1,1) * gfunc
    #tfunc = gfunc * theta
    gfunc_tensors = np.zeros([3,3,ne,nt],dtype='complex_')
    tfunc = np.zeros([3,3,ne,nt],dtype='complex_')
    for x in range(3):
        for y in range(3):
            for e in range(ne):
                gfunc_tensors[x,y,e] = dip_matrix[x,y,e] * gfunc[e]
                tfunc[x,y,e] = gfunc_tensors[x,y,e] * theta

    for e in range(ne):
        dump = np.zeros(nt,dtype="complex_")
        for a in range(vibquanta_max+1):
            dump += FCfunc(disps[e,j-1]/np.sqrt(2),a,1) * \
                    FCfunc(disps[e,j-1]/np.sqrt(2),a,1) * \
                    np.exp(-1.0j * (a-1.0) * vibfreqs[j-1] * t_list)
        tfunc[:,:,e] *= dump

    timelist = bndwth * np.add.reduce(tfunc,axis=2)

    alpha = np.zeros([3,3,nt], dtype="complex_")
    alpha_data = np.zeros([3,3,nOmegaL],dtype="complex_")
    for x in range(3):
        for y in range(3):
            timelist[x,y] = np.concatenate(
                (timelist[x,y,nt/2-1:], timelist[x,y,:nt/2-1])
            )
            timelist[x,y] = np.fft.fft(timelist[x,y]) / np.sqrt(nt)
            timelist[x,y,1:] = timelist[x,y,:0:-1]
            alpha[x,y] = np.concatenate(
                (timelist[x,y,nt/2+2:], timelist[x,y,:nt/2+2])
            )

            tck_r = interpolate.splrep(omega_list,np.real(alpha[x,y]))
            tck_i = interpolate.splrep(omega_list,np.imag(alpha[x,y]))

            alpha_data[x,y] = interpolate.splev(OmegaL,tck_r) + \
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
    #gfunc = dipoles.reshape(-1,1) * gfunc
    #tfunc = gfunc * theta
    dip_matrix = np.zeros([3,3,ne])
    for e in range(ne):
        dip_matrix[:,:,e] = np.outer(dipvecs[e],dipvecs[e])
    gfunc_tensors = np.zeros([3,3,ne,nt],dtype='complex_')
    tfunc = np.zeros([3,3,ne,nt],dtype='complex_')
    for x in range(3):
        for y in range(3):
            for e in range(ne):
                gfunc_tensors[x,y,e] = dip_matrix[x,y,e] * gfunc[e]
                tfunc[x,y,e] = gfunc_tensors[x,y,e] * theta

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
        tfunc[:,:,e] *= dump

    timelist = bndwth * np.add.reduce(tfunc,axis=2)

    alpha = np.zeros([3,3,nt], dtype="complex_")
    alpha_data = np.zeros([3,3,nOmegaL], dtype='complex_')
    for x in range(3):
        for y in range(3):
            timelist[x,y] = np.concatenate(
                (timelist[x,y,nt/2-1:], timelist[x,y,:nt/2-1])
            )
            timelist[x,y] = np.fft.fft(timelist[x,y]) / np.sqrt(nt)
            timelist[x,y,1:] = timelist[x,y,:0:-1]
            alpha[x,y] = np.concatenate(
                (timelist[x,y,nt/2+2:], timelist[x,y,:nt/2+2])
            )

            tck_r = interpolate.splrep(omega_list,np.real(alpha[x,y]))
            tck_i = interpolate.splrep(omega_list,np.imag(alpha[x,y]))

            alpha_data[x,y] = interpolate.splev(OmegaL,tck_r) + \
                1.0j * interpolate.splev(OmegaL,tck_i)
#    print interpolate.splev(energies[0],tck_r) + \
#            1.0j * interpolate.splev(energies[0],tck_i)
    return alpha_data

def Cal_alpha(OmegaL,SYSTEM,PARAM):
    vibfreqs = SYSTEM['vibfreqs']
    nmodes = len(vibfreqs)
    nstates = nmodes + 1
    nex = len(OmegaL)
    alpha = np.zeros([nstates,nstates,3,3,nex],dtype="complex_")
    n_elem = (nstates + 1) * nstates / 2
    print '%d alpha elements in calculation...' % n_elem
    if interactive: sys.stdout.write('alpha element:            ')
    ialpha = 1
    backspace = '\b'*11
    for n in range(nstates):
        for m in range(n,nstates):
            if interactive:
                sys.stdout.write("%s%5d/%5d" % (backspace,ialpha,n_elem))
                sys.stdout.flush()
            if n==0 and m==0:
                alpha[n,m] = ae1(OmegaL,SYSTEM,PARAM)
            elif n==0 and m!=0:
                alpha[n,m] = ae2(m,OmegaL,SYSTEM,PARAM)
            elif n!=0 and n==m:
                alpha[n,m] = ae5(n,OmegaL,SYSTEM,PARAM)
            else:
                alpha[n,m] = ae6(n,m,OmegaL,SYSTEM,PARAM)
            ialpha += 1

    for i in range(nstates):
        for j in range(i):
            for e in range(nex):
                alpha[i,j,:,:,e] = np.transpose(alpha[j,i,:,:,e])
    return alpha

if __name__ == '__main__':
    PARAM = initialize_param()
    SYSTEM = initialize_system(PARAM)
    nmodes = len(SYSTEM['vibfreqs'])
    OmegaL = PARAM['PU_pulses_list']
    print 'Number of excitation lasers: ', len(OmegaL)
    print 'Excitation lasers:'
    print OmegaL
    dipvecs = SYSTEM['dipvecs']

    interactive = True

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
    print "\ncalculating alpha done."

    # write the alpha tensor to npy file
    # now alpha.shape is (nmodes+1, nmodes+1, 3, 3, nOmegaL)
    # new alpha.shape is (nOmegaL, nmodes+1, nmodes+1, 3, 3)
    new_alpha = np.zeros([len(OmegaL),nmodes+1,nmodes+1,3,3],dtype='complex_')
    for e in range(len(OmegaL)):
        new_alpha[e] = alpha[:,:,:,:,e]
    np.save('p-alpha.npy',new_alpha)
    print 'alpha tensors saved to p-alpha.npy'
