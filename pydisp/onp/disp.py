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
import numpy as np
from constants import *
from parse import *
#import scitools.numpyutils as scinp
#==============================================================================
def normalize(v):
    N = 1.0 / np.sqrt(np.dot(v,v))
    return v * N, N

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

parser.add_argument(
    "-s", "--scale",
    help = "scaling factor to calculated frequencies",
    default = 2.0,
    type = float
)

args = parser.parse_args()

gs_fname = args.freqfile
ex_fname = args.excitedfile
ofname = args.output
linear = args.linear
factor = args.scale

if factor > 1.5:
    print "Warning: using default freqency scale factor 2.0, which should be"
    print "         an unreasonable value..."
    print "         set the factor by using -s or --scale"
    raise SystemExit

print "= using ground state vibration logfile %s" % gs_fname
print "= using excited state gradient logfile %s" % ex_fname
print "= linear molecule: ", linear
print "= freqency scaling factor: ", factor
print 
#=========================================================================

#=========================================================================
# get Number of atoms 
N = get_num_atoms(gs_fname)
coords = get_coordinates(gs_fname, N)
coords *= AUTOA
print "Number of atoms: ", N
if linear:
    nmodes = 3 * N - 5
else:
    nmodes = 3 * N - 6
print "Number of vibration modes: ", nmodes


# get atomic masses
atm_mass = get_atomic_masses(gs_fname, N)
# atm_mass in atomic mass unit
#print atm_mass
atm_mass *= AMUTOAU
M_vec = np.zeros([3*N,1])
for i in range(3*N):
    M_vec[i] = atm_mass[i/3]

# get frequencies and normal modes
# freq in cm-1
omega_org, L = get_freq_modes(gs_fname, N, nmodes)
omega = omega_org * factor
#print freq
#print L[:,-1]
# convert freq from cm-1 to frequency atomic unit
# 1 au  = 4.13E16 sec-1
freq = omega * 100. * C0 / FREQAU * (2*PI)
freq = freq.reshape(-1,1)


# get gradients of ground state and excited states
# here we got gradients in atomic units (Hartree/Bohr)
gs_grad = get_gradient(gs_fname, N)
ex_grad = get_gradient(ex_fname, N)
#print gs_grad
#print ex_grad
#print ex_grad - gs_grad

diff_grad = ex_grad - gs_grad
diff_grad = diff_grad.reshape(-1,1)

# calculate the dimensionless displacements
# According to formulas in Refs:
#   J. Guthmuller, B. Champagne, JCP, 127, 164507(2007)
#   J. Guthmuller, JCTC, 7, 1082(2011)
#   J. Guthmuller, PCCP, 12, 14812(2010)
#
#   dQ = (\partial E / \partial Q) = L^T . M^{-1/2} . (\partial E / \partial x)
#   Delta = - 1 / (sqrt(\hbar) *  )
#
grad_Q = (1.0 / np.sqrt(M_vec)) * diff_grad
grad_Q = np.dot(np.transpose(L), grad_Q)
#print "grad_Q", grad_Q.shape
#print "freq", freq.shape
#raise SystemExit

#print dQ
D = freq**(-1.5) * grad_Q
print "="*50
print
for i in range(nmodes):
    print "%12.2f%12.2f%12.4f" % (omega_org[i], omega[i], D[i])


#fh = open(ofname, "w")
#for i in range(nmodes):
#    fh.write("%12.4f\t%18.6E\n" % (omega[i],D[i]))
#fh.close()

