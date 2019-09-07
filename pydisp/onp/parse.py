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
import inspect
import numpy as np
#==============================================================================

#=========================================================================
# helper functions parsing Gaussian 09 outputs
#=========================================================================
def get_num_atoms(gs_fname):
    myname = inspect.stack()[0][3]
    try:
        fh = open(gs_fname, "r")
    except:
        print "IOError: can not open file %s (in %s)" % (gs_fname, myname)
        raise SystemExit
    for l in fh:
        if "Input orientation" in l:
            break
    for i in range(4):
        l = fh.next()
    na = 0
    for l in fh:
        if len(l.split()) == 6:
            na += 1
        else:
            break
    fh.close()
    return na

def get_coordinates(gs_fname, N):
    myname = inspect.stack()[0][3]
    try:
        fh = open(gs_fname, "r")
    except:
        print "IOError: can not open file %s (in %s)" % (gs_fname, myname)
        raise SystemExit
    for l in fh:
        if "Input orientation" in l:
            break
    for i in range(4):
        l = fh.next()
    coord = np.zeros([N,3])
    for ia in range(N):
        l = fh.next().split()[3:]
        coord[ia] = [float(x) for x in l]
    fh.close()
    return coord


def get_atomic_masses(gs_fname, N):
    myname = inspect.stack()[0][3]
    masses = np.zeros(N)
    try:
        fh = open(gs_fname, "r")
    except:
        print "IOError: can not open file %s (in %s)" % (gs_fname, myname)
        raise SystemExit
    nblock = (N-1) / 10 + 1

    for l in fh:
        if "Isotopes and Nuclear Properties" in l:
            break
    for i in range(3):
        l = fh.next()
    for iblock in range(nblock):
        fh.next()
        fh.next()
        l = fh.next()
        last = min((iblock+1)*10, N)
        masses[iblock*10:last] = np.array([float(x) for x in l.split()[1:]])
        for i in range(5):
            fh.next()
    return masses

def get_hessian(gs_fname, N):
    myname = inspect.stack()[0][3]
    hessian = np.zeros([3*N,3*N])
    try:
        fh = open(gs_fname, "r")
    except:
        print "IOError: can not open file %s (in %s)" % (gs_fname, myname)
        raise SystemExit
    
    s = "Force constants in Cartesian coordinates:"
    nblock = (3*N-1) / 5 + 1
    
    for l in fh:
        if s in l:
            break

    for iblock in range(nblock):
        l = fh.next()
        for ir in range(iblock*5,3*N):
            l = fh.next().replace("D", "E")
            l = l.split()[1:]
            ne = len(l)
            hessian[ir,iblock*5:iblock*5+ne] = np.array([float(x) for x in l])
    fh.close()
    for i in range(3*N):
        for j in range(i,3*N):
            hessian[i,j] = hessian[j,i]
    return hessian

def get_gradient(fname, N):
    myname = inspect.stack()[0][3]
    grad = np.zeros([N,3])
    try:
        fh = open(fname, "r")
    except:
        print "IOError: can not open file %s (in %s)" % (fname, myname)
        raise SystemExit

    string = 'Center     Atomic                   Forces (Hartrees/Bohr)'
    
    for l in fh:
        if string in l:
            break

    fh.next()
    fh.next()
    for i in range(N):
        l = fh.next()
        l = l.split()[2:]
        grad[i,:] = np.array([float(x) for x in l])
    fh.close()
    return grad
    
def get_freq_modes(gs_fname, N, nmodes):
    myname = inspect.stack()[0][3]
    freq = np.zeros(nmodes)
    modes = np.zeros([3*N,nmodes])
    nblock = (nmodes - 1) / 5 + 1
    try:
        fh = open(gs_fname, "r")
    except:
        print "IOError: can not open file %s (in %s)" % (gs_fname, myname)
        raise SystemExit
    for l in fh:
        if "Full mass-weighted force" in l:
            break
    for i in range(6):
        fh.next()
    for iblock in range(nblock):
        fh.next()
        fh.next()
        last = min(nmodes, (iblock+1) * 5)
        l = fh.next().split()[2:]
        freq[iblock*5:last] = np.array([float(x) for x in l])
        for i in range(4):
            fh.next()
        for il in range(3*N):
            l = fh.next().split()[3:]
            modes[il,iblock*5:last] = np.array([float(x) for x in l])
    fh.close()
    return freq, modes


