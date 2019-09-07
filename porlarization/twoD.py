#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==============================================================================
# Module documentation
"""
FileName	: oneD.py
Purpose		: 1D polarized stimulated Raman spectra 
Author		: Hao Ren
Version		: 0.1
Date		: Sep 20, 2012
"""
#==============================================================================

#==============================================================================
# Module imports
import numpy as np
import math
import time
import sys
from common import *
#==============================================================================

def def_M6():
    M6 = [
        [16, -5, -5, -5, 2, 2, -5, 2, 2, 2, 2, -5, 2, 2, -5],
        [-5, 16, -5, 2, -5, 2, 2, 2, -5, -5, 2, 2, 2, -5, 2],
        [-5, -5, 16, 2, 2, -5, 2, -5, 2, 2, -5, 2, -5, 2, 2],
        [-5, 2, 2, 16, -5, -5, -5, 2, 2, 2, -5, 2, 2, -5, 2],
        [2, -5, 2, -5, 16, -5, 2, -5, 2, -5, 2, 2, 2, 2, -5],
        [2, 2, -5, -5, -5, 16, 2, 2, -5, 2, 2, -5, -5, 2, 2],
        [-5, 2, 2, -5, 2, 2, 16, -5, -5, -5, 2, 2, -5, 2, 2],
        [2, 2, -5, 2, -5, 2, -5, 16, -5, 2, -5, 2, 2, 2, -5],
        [2, -5, 2, 2, 2, -5, -5, -5, 16, 2, 2, -5, 2, -5, 2],
        [2, -5, 2, 2, -5, 2, -5, 2, 2, 16, -5, -5, -5, 2, 2],
        [2, 2, -5, -5, 2, 2, 2, -5, 2, -5, 16, -5, 2, -5, 2],
        [-5, 2, 2, 2, 2, -5, 2, 2, -5, -5, -5, 16, 2, 2, -5],
        [2, 2, -5, 2, 2, -5, -5, 2, 2, -5, 2, 2, 16, -5, -5],
        [2, -5, 2, -5, 2, 2, 2, 2, -5, 2, -5, 2, -5, 16, -5],
        [-5, 2, 2, 2, -5, 2, 2, -5, 2, 2, 2, -5, -5, -5, 16]
    ]
    return np.array(M6)/210.0

def rotation_matrix(theta):
    R = np.array(
        [[math.cos(theta), -1.0*math.sin(theta), 0.],
         [math.sin(theta), math.cos(theta), 0,],
         [0.,0.,1.]]
    )
    return R

def load_alpha(conf):
    p1 = int(conf[0])-1
    p2 = int(conf[1])-1
    p3 = int(conf[2])-1

    alpha = np.load('p-alpha.npy')
    a1 = alpha[p1]
    a2 = alpha[p2]
    a3 = alpha[p3]
    return a1, a2, a3

def parse_vecs(vecs):
    d = [vecs[i:i+1] for i in range(len(vecs))]
    v = [[0,0,0] for i in d]
    ma = math.acos(1./math.sqrt(3.))
    v_ma = np.dot(rotation_matrix(ma),[1,0,0])
    if vecs == 'AAA':
        v[0] = [1.0, 0.0, 0.0]
        v[1] = [0.353397, -0.609906, -0.709313]
        v[2] = [0.171867, 0.000, -0.985120]
        return v
    if vecs == 'BBB':
        v[0] = [1.000, 0.000, 0.000]
        v[1] = [0.736055, 0.479544, 0.477766]
        v[2] = [0.798203, 0.000, 0.602388]
        return v
    for i in range(len(d)):
        direction = d[i]
        if direction == 'H':
            v[i] = np.array([1,0,0])
        elif direction == 'V':
            v[i] = np.array([0,1,0])
        elif direction == 'M':
            # magic angle
            v[i] = np.array(v_ma)
        else:
            print "Unaccepted vector direction: %s" % direction
            raise SystemExit
    return v

def orfactor1D(e1,e2,M4):
    f1, f2, f3, f4 = e1, e1, e2, e2
    field = [
        np.dot(f1,f2) * np.dot(f3,f4),
        np.dot(f1,f3) * np.dot(f2,f4),
        np.dot(f1,f4) * np.dot(f2,f3)
    ]
    return np.dot(M4,field)

def orfactor2D(e1,e2,e3,M6):
    f1,f2,f3,f4,f5,f6 = e1, e1, e2, e2, e3, e3
    field = [
        np.dot(f1,f2) * np.dot(f3,f4) * np.dot(f5,f6),
        np.dot(f1,f2) * np.dot(f3,f5) * np.dot(f4,f6),
        np.dot(f1,f2) * np.dot(f4,f5) * np.dot(f3,f6),
        np.dot(f1,f3) * np.dot(f2,f4) * np.dot(f5,f6),
        np.dot(f1,f3) * np.dot(f2,f5) * np.dot(f4,f6),
        np.dot(f1,f3) * np.dot(f4,f5) * np.dot(f2,f6),
        np.dot(f2,f3) * np.dot(f1,f4) * np.dot(f5,f6),
        np.dot(f1,f4) * np.dot(f2,f5) * np.dot(f3,f6),
        np.dot(f1,f4) * np.dot(f3,f5) * np.dot(f2,f6),
        np.dot(f2,f3) * np.dot(f1,f5) * np.dot(f4,f6),
        np.dot(f2,f4) * np.dot(f1,f5) * np.dot(f3,f6),
        np.dot(f3,f4) * np.dot(f1,f5) * np.dot(f2,f6),
        np.dot(f2,f3) * np.dot(f4,f5) * np.dot(f1,f6),
        np.dot(f2,f4) * np.dot(f3,f5) * np.dot(f1,f6),
        np.dot(f3,f4) * np.dot(f2,f5) * np.dot(f1,f6),
    ]
    return np.dot(M6,field)

def ampfunc(a1,a2,a3,orfactor):
    Tr = np.trace
    Transpose = np.transpose
    dump = [
        Tr(a1) * Tr(a2) * Tr(a3),
        Tr(a1) * Tr(np.dot(Transpose(a2),a3)),
        Tr(a1) * Tr(np.dot(a3,a2)),
        Tr(np.dot(a1,Transpose(a2))) * Tr(a3),
        Tr(np.dot(np.dot(Transpose(a1),a2),Transpose(a3))),
        Tr(np.dot(np.dot(Transpose(a1),a2),a3)),
        Tr(np.dot(a1,a2)) * Tr(a3),
        Tr(np.dot(np.dot(Transpose(a1),Transpose(a2)),Transpose(a3))),
        Tr(np.dot(np.dot(Transpose(a1),Transpose(a2)),a3)),
        Tr(np.dot(np.dot(a1,a2),Transpose(a3))),
        Tr(np.dot(np.dot(a1,Transpose(a2)),Transpose(a3))),
        Tr(a2) * Tr(np.dot(Transpose(a1),a3)),
        Tr(np.dot(np.dot(a1,a2),a3)),
        Tr(np.dot(np.dot(a1,Transpose(a2)),a3)),
        Tr(a2) * Tr(np.dot(a1,a3))
    ]
    return np.dot(orfactor,dump)

def prepare_amp_freq_S1(gsEnergies,a1,a2,a3,orfactor):
    conj = np.conjugate
    nmodes = len(gsEnergies) -1
    Amp1 = np.zeros([nmodes+1,nmodes+1],dtype='complex_')
    freq1 = np.zeros([nmodes+1,nmodes+1,2])
    for c in range(nmodes+1):
        for cp in range(nmodes+1):
            freq1[c,cp] = [gsEnergies[c]-gsEnergies[0],gsEnergies[c]-gsEnergies[cp]]
            if c == cp: continue
            if c == 0: continue
            Amp1[c,cp] = ampfunc(a1[c,0],conj(a2[0,cp]),a3[cp,c],orfactor)
    return Amp1, freq1

def prepare_amp_freq_S2(gsEnergies,a1,a2,a3,orfactor):
    conj = np.conjugate
    nmodes = len(gsEnergies) - 1
    Amp2 = np.zeros([nmodes+1,nmodes+1],dtype='complex_')
    freq2 = np.zeros([nmodes+1,nmodes+1,2])
    for c in range(nmodes+1):
        for cp in range(nmodes+1):
            freq2[c,cp] = [gsEnergies[0]-gsEnergies[c],gsEnergies[0]-gsEnergies[cp]]
            if c == 0: continue
            if cp == 0: continue
            Amp2[c,cp] = ampfunc(conj(a1[0,c]),conj(a2[c,cp]),a3[cp,0],orfactor)
    return Amp2, freq2

def prepare_amp_freq_S3(gsEnergies,a1,a2,a3,orfactor):
    conj = np.conjugate
    nmodes = len(gsEnergies) - 1
    Amp3 = np.zeros([nmodes+1,nmodes+1],dtype='complex_')
    freq3 = np.zeros([nmodes+1,nmodes+1,2])
    for c in range(nmodes+1):
        for cp in range(nmodes+1):
            freq3[c,cp] = [gsEnergies[0]-gsEnergies[c],gsEnergies[cp]-gsEnergies[c]]
            if c == cp: continue
            if c == 0: continue
            Amp3[c,cp] = ampfunc(conj(a1[0,c]),a2[cp,0],a3[c,cp],orfactor)
    return Amp3, freq3

def prepare_amp_freq_S4(gsEnergies,a1,a2,a3,orfactor):
    conj = np.conjugate
    nmodes = len(gsEnergies) - 1
    Amp4 = np.zeros([nmodes+1,nmodes+1],dtype='complex_')
    freq4 = np.zeros([nmodes+1,nmodes+1,2])
    for c in range(nmodes+1):
        for cp in range(nmodes+1):
            freq4[c,cp] = [gsEnergies[c]-gsEnergies[0],gsEnergies[cp]-gsEnergies[0]]
            if c == 0: continue
            if cp == 0: continue
            Amp4[c,cp] = ampfunc(a1[c,0],a2[cp,c],a3[0,cp],orfactor)
    return Amp4, freq4


def cal_Signal(amp,freq,Omega1,Omega2,gammaG):

    S = np.zeros([nOmega2,nOmega1],dtype='complex_')
    w01 = freq[:,:,0]
    w02 = freq[:,:,1]
    ramp = np.real(amp)
    iamp = np.imag(amp)
    A1 = ramp * gammaG**2
    A2 = iamp * gammaG * (w01 + w02)
    A3 = ramp * w01 * w02
    AA1 = A1 + A2 - A3
    for i in range(nOmega2):
        for j in range(nOmega1):
            w1 = Omega1[j]
            w2 = Omega2[i]
            A4 = ramp * gammaG * w1
            A5 = iamp * w02 * w1
            A6 = ramp * gammaG * w2
            A7 = iamp * w01 * w2
            A8 = ramp * w1 * w2
            BB = (gammaG**2 + w01*w01 - 2.0j * gammaG * w1 - w1*w1) * (
                gammaG**2 + w02*w02 - 2.0j * gammaG * w2 - w2*w2)
            AA = AA1 - 1.0j * (A4 + A5 + A6 + A7) - A8
            S[i,j] = np.sum(AA/BB) 
    S /= (2.0*math.pi)
    return S

if __name__ == '__main__':
    PARAM = initialize_param()
    SYSTEM = initialize_system(PARAM)

    beta = PARAM['beta']
    sigma = PARAM['sigma']
    gammaE = PARAM['Gamma']
    gammaG = PARAM['Gamma_g']
    basetime = PARAM['basetime']
    #print 'beta', beta
    #print 'sigma', sigma/basetime
    #print 'gammaE', gammaE
    #print 'gammaG', gammaG

    Omega1_min = PARAM['POR2D_Omega1_min']
    Omega1_max = PARAM['POR2D_Omega1_max']
    nOmega1 = PARAM['POR2D_nOmega1']
    Omega1 = np.linspace(Omega1_min,Omega1_max,nOmega1)
    Omega2_min = PARAM['POR2D_Omega2_min']
    Omega2_max = PARAM['POR2D_Omega2_max']
    nOmega2 = PARAM['POR2D_nOmega2']
    Omega2 = np.linspace(Omega2_min,Omega2_max,nOmega2)

    vibfreqs = SYSTEM['vibfreqs']
    nmodes = len(vibfreqs)

    conf = PARAM['POR2D_PULSE_CONF']
    a1, a2, a3 = load_alpha(conf)

    vecs = PARAM['POR2D_PULSE_VEC']
    v = parse_vecs(vecs)
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    print v1
    print v2
    print v3
    raise SystemExit

    M6 = def_M6()

    orfactor = orfactor2D(v1,v2,v3,M6)

    gsEnergies = np.concatenate(
        ([0], vibfreqs)
    )

    sys.stdout.write('preparing amplitudes and frequencies... ')
    t1 = time.clock()
    amp1,freq1 = prepare_amp_freq_S1(gsEnergies,a1,a2,a3,orfactor)
    amp2,freq2 = prepare_amp_freq_S2(gsEnergies,a1,a2,a3,orfactor)
    amp3,freq3 = prepare_amp_freq_S3(gsEnergies,a1,a2,a3,orfactor)
    amp4,freq4 = prepare_amp_freq_S4(gsEnergies,a1,a2,a3,orfactor)
    sys.stdout.write('Done! used time: %f\n' % (time.clock()-t1))

    print('\ncalculating S1... ')
    t1 = time.clock()
    S1 = cal_Signal(amp1,freq1,Omega1,Omega2,gammaG)
    print('Done! used time: %f\n' % (time.clock()-t1))

    print('\ncalculating S2... ')
    t1 = time.clock()
    S2 = cal_Signal(amp2,freq2,Omega1,Omega2,gammaG)
    print('Done! used time: %f\n' % (time.clock()-t1))

    print('\ncalculating S3... ')
    t1 = time.clock()
    S3 = cal_Signal(amp3,freq3,Omega1,Omega2,gammaG)
    print('Done! used time: %f\n' % (time.clock()-t1))

    print('\ncalculating S4... ')
    t1 = time.clock()
    S4 = cal_Signal(amp4,freq4,Omega1,Omega2,gammaG)
    print('Done! used time: %f\n' % (time.clock()-t1))

    S = S1 + S2 + S3 + S4

    basename = 'POR2D-%s-%s' % (vecs,conf)
    # whole signal information save to npy file
    np.save('%s.npy'%basename, S)
    print("2D data saved to file %s.npy"% basename)



