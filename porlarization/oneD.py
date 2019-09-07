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
    alpha = np.load('p-alpha.npy')
    a1 = alpha[p1]
    a2 = alpha[p2]
    return a1, a2

def parse_vecs(vecs):
    d = [vecs[i:i+1] for i in range(len(vecs))]
    v = [[0,0,0] for i in d]
    ma = math.acos(1./math.sqrt(3.))
    v_ma = np.dot(rotation_matrix(ma),[1,0,0])
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

def ampfunc(a1,a2,orfactor):
    dump = [
        np.trace(a1) * np.trace(a2),
        np.trace(np.dot(a1,np.transpose(a2))),
        np.trace(np.dot(a1,a2))
    ]
    return np.dot(orfactor,dump)



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

    OmegaS_min = PARAM['POR1D_OmegaS_min']
    OmegaS_max = PARAM['POR1D_OmegaS_max']
    nOmegaS = PARAM['POR1D_nOmegaS']
    OmegaS = np.linspace(OmegaS_min,OmegaS_max,nOmegaS)
    S = np.zeros(nOmegaS, dtype='complex_')
    
    vibfreqs = SYSTEM['vibfreqs']
    nmodes = len(vibfreqs)
    newfreqs = np.concatenate(
        (vibfreqs, -1*vibfreqs)
    )

    conf = PARAM['POR1D_PULSE_CONF']
    a1, a2 = load_alpha(conf)

    vecs = PARAM['POR1D_PULSE_VEC']
    v = parse_vecs(vecs)
    v1 = v[0]
    v2 = v[1]
    #print vecs

    M4 = np.array(
        [[4, -1, -1],
         [-1, 4, -1],
         [-1, -1, 4]]
    ) / 30.0

    orfactor = orfactor1D(v1,v2,M4)

    Amp = np.zeros(nmodes*2, dtype='complex_')
    for i in range(nmodes):
        Amp[i] = ampfunc(a1[0,i+1],a2[i+1,0],orfactor)
        Amp[i+nmodes] = ampfunc(np.conjugate(a1[0,i+1]),a2[i+1,0],orfactor)
    rAmp = np.real(Amp)
    iAmp = np.imag(Amp)

    for i in range(nOmegaS):
        for n in range(2*nmodes):
            S[i] += (rAmp[n] * (gammaG - 1.0j * OmegaS[i]) + 
                     iAmp[n] * newfreqs[n] ) / ( gammaG**2 - 2.0j *
                    gammaG * OmegaS[i] - OmegaS[i]**2 + newfreqs[n]**2)
    S *= (-1.0/math.sqrt(2*math.pi))

    basename = 'POR1D-%s-%s' % (vecs,conf)
    textfile = '%s.dat' % basename
    fh = open(textfile,'w')
    fh.write('# omega_s     intensity (abs(S))\n')
    for i in range(nOmegaS):
        fh.write('%10.2f%15.5e\n' % (OmegaS[i],abs(S[i])))
    fh.close()
    
    # whole signal information save to npy file
    np.save('%s.npy'%basename, S)



