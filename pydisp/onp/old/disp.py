#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==============================================================================
# Module documentation
"""
FileName	: disp.py
Purpose		: 
Author		: Hao Ren
Version		: 0.1
Date		: , 2011
"""
#==============================================================================

#==============================================================================
# Module imports
import argparse
import inspect
import numpy as np
#==============================================================================

# parse arguments for input file names
parser = argparse.ArgumentParser()
parser.add_argument(
    "-gf", "--freqfile",
    default = "freq.log",
    help="G09 ground state vibration analysis logfile"
)
parser.add_argument(
    "-ef","--excitedfile", 
    default = "force.log",
    help="G09 excited state gradients logfile",
)
parser.add_argument(
    "-o", "--output",
    default = "disp.dat",
    help = "output file contains dimensionless displacements"
)

parser.add_argument(
    "-l","--linear",
    help="linear molecule (geometry)",
    action = "store_true"
)

args = parser.parse_args()

gs_fname = args.freqfile
ex_fname = args.excitedfile
ofname = args.output
linear = args.linear

print "= using ground state vibration logfile %s" % gs_fname
print "= using excited state gradient logfile %s" % ex_fname
print "= linear molecule: ", linear
print 
#=========================================================================

#=========================================================================
# some constants
AUTOA = 0.529177249 # Bohr to angstrom
PI = 3.141592653589793238 
hbar = 1.054571628E-34 
AMUTOAU = 1822.8889 # AMU to au
C0 = 299792458. # light speed
WAVNUM2AU= 100.0*C0/6.57969E15
#=========================================================================

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

def get_gradient(fname, N):
    myname = inspect.stack()[0][3]
    try:
        fh = open(fname, "r")
    except:
        print "IOError: can not open file %s (in %s)" % (fname, myname)
        raise SystemExit
    string = "Center     Atomic                   Forces"
    for l in fh:
        if string in l:
            break
    fh.next()
    fh.next()
    force = np.zeros([N,3])
    for i in range(N):
        l = fh.next().split()[2:]
        force[i] = np.array([float(x) for x in l])
    fh.close()
    return force

def get_hessian(fname,N):
    myname = inspect.stack()[0][3]
    try:
        fh = open(fname,"r")
    except:
        print "IOError: can not open file %s (in %s)" % (fname, myname)
        raise SystemExit
    string = "Force constants in Cartesian coordinates"
    for l in fh:
        if string in l:
            break
    nblock = (3*N-1) / 5 + 1
    hessian = np.zeros([3*N,3*N])
    for iblock in range(nblock):
        fh.next()
        for i in range(iblock*5,3*N):
            l = fh.next().replace("D","E").split()
            nc = len(l) - 1
            hessian[i,iblock*5:iblock*5+nc] = np.array([float(x) for x in l[1:]])
    return hessian
            


#=========================================================================

#=========================================================================
#=========================================================================
# get Number of atoms 
N = get_num_atoms(gs_fname)
print "Number of atoms: ", N
if linear:
    nmodes = 3 * N - 5
else:
    nmodes = 3 * N - 6
print "Number of vibration modes: ", nmodes


# get atomic masses
atm_mass = get_atomic_masses(gs_fname, N)
# atm_mass in atomic mass unit
atm_mass *= AMUTOAU
M = np.zeros([3*N,1])
for i in range(3*N):
    M[i] = atm_mass[i/3]

# read Hessian Matrix from gs_fname
hessian = get_hessian(gs_fname, N)
for i in range(3*N):
    for j in range(i,3*N):
        hessian[i,j] = hessian[j,i]
weighted_hessian = hessian / np.sqrt(np.outer(M,M.transpose()))
w, L = np.linalg.eig(weighted_hessian)
idx = w.argsort()
w = w[idx][6:]
L = L.T[idx][6:].T

freq = np.sqrt(w) * 4.13E16 * 0.01 / C0 / (2. * PI)

L_mw = L


# get frequencies and normal modes
# freq in cm-1
freq, L_car = get_freq_modes(gs_fname, N, nmodes)

#print np.sum(l*l)
#raise SystemExit

# get gradients of ground state and excited states
# here we got gradients in atomic units (Hartree/Bohr)
gs_grad = get_gradient(gs_fname, N)
ex_grad = get_gradient(ex_fname, N)

diff_grad = ex_grad - gs_grad
diff_grad = diff_grad.reshape(-1,1)

# calculate the dimensionless displacements
# According to formulas in Refs:
#   J. Guthmuller, B. Champagne, JCP, 127, 164507(2007)
#   J. Guthmuller, JCTC, 7, 1082(2011)
#
#   \Delta = \sqrt{\omega / hbar} * dQ
#   dQ = - 1 / omega^2 * (\partial E / \partial Q)
#   (\partial E / \partial Q) = L^T * M^{-1/2} * (\partial E / \partial x)
#
#print L[:,8]
#print np.dot(L[:,8].transpose(),np.dot(M_msqrt,diff_grad))
grad_Q = np.dot(np.transpose(L_mw), 1.0/np.sqrt(M)*diff_grad)
grad_Q = grad_Q.reshape(-1,1)

# convert freq from cm-1 to frequency atomic unit
# 1 au  = 4.13E16 sec-1
omega = freq
freq = freq * 100. * C0 / 4.13E16
freq = freq.reshape(-1,1)
dQ = -1.0 / (freq*freq) * grad_Q
#print dQ
D = np.sqrt(freq) * dQ
print D
print np.sum(L[:,8]*L[:,8])


#fh = open(ofname, "w")
#for i in range(nmodes):
#    fh.write("%12.4f\t%18.6E\n" % (omega[i],D[i]))
#fh.close()

