#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==============================================================================
# Module documentation
"""
FileName	: .py
Purpose		: 
Author		: Hao Ren
Version		: 0.1
Date		: , 2011
"""
#==============================================================================

#==============================================================================
# Module imports
import numpy as np
import sys
import math
#==============================================================================

try:
    script, iex = sys.argv
except:
    print 'Usage: %s iex' % sys.argv[0]
    raise SystemExit

iex = int(iex) - 1

alpha = np.load('p-alpha.npy')

nmodes = alpha.shape[1]

theta = range(0,181,5)
phi = range(0,361,5)
ntheta = len(theta)
nphi = len(phi)

for im in range(1,nmodes):
    s = np.zeros([ntheta,nphi],dtype='complex_')
    for i in range(ntheta):
        for j in range(nphi):
            t = theta[i] / 180. * math.pi
            p = phi[j] / 180. * math.pi
            e1 = np.array([math.sin(t)*math.cos(p), 
                  math.sin(t)*math.sin(p),
                math.cos(t)
                ],dtype='complex_')
            # parallel polarization for incident and scattering beams
            e2 = np.array([d for d in e1],dtype='complex_')
            a = alpha[iex,0,im]
            s[i,j] = np.dot(
                np.dot(e1.reshape(1,-1),a), e2
            )[0]

    s_f = 'ang-%d-%d.npy' % (iex+1,im)
    np.save(s_f, s)



