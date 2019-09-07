#!/usr/bin/env python

#==============================================================================
# Module documentation
"""
FileName	: common.py
Purpose		: Common functions for absorption, CWRaman, 1D and 2D stimulated Raman
              Calculations
Author		: Hao Ren (translated from Jason's mathematica notebook)
Version		: 1.0
Date		: Jan 23, 2012
Description : All functions would be initialized by this module. Some of the parameters
              are redundant for one calculation but necessary for another.
-------------------------------------------------------------------------------
Reversion   : 1.1
Date        : Jan 24, 2012
Description : add parameter values for 2D calculations. add pulse control.
"""
#==============================================================================

#==============================================================================
# Module imports
import numpy as np
import time
import sys
#==============================================================================


def initialize_param():

    f_energy_dipole = 'energies_dipoles.inp'
    f_disp = 'disp.inp'
    
    basetime = 100
    num = 14
    domega = 10.0
    sigma = 0.005  # 5 fs in the units used here
    
    Gamma = 100.
    Gamma_g = 10.0
    beta = 1/200.
    vibquanta_max = 6 # |0>...|6>, 7 states in total
    
    ABS_OmegaL_min,ABS_OmegaL_max,ABS_nOmegaL = (30000, 50000, 2001)
    
    CW_read_alpha = False
    CW_write_alpha = True
    CW_cal_2DCW = False
    CW_save_txt = True
    CW_save_npy = True
    
    CW_OmegaS_min,CW_OmegaS_max,CW_nOmegaS = (600.0, 1800.0, 1201)
    CW_OmegaL_min,CW_OmegaL_max,CW_nOmegaL = (-4000, 4000, 2000)
    
    PU1d_OmegaS_min,PU1d_OmegaS_max,PU1d_nOmegaS = (600.0, 1800.0, 1201)
    
    TD_read_alpha = False
    TD_Omega1_min,TD_Omega1_max,TD_nOmega1 = (0.0, 1900.0, 191)
    TD_Omega2_min,TD_Omega2_max,TD_nOmega2 = (-1900.0, 1900.0, 381)
    TD_configuration = 111

    POR1D_OmegaS_min = 600.0
    POR1D_OmegaS_max = 1800.0
    POR1D_nOmegaS = 1201
    
    POR1D_PULSE_CONF = 11
    POR1D_PULSE_VEC = 'HH'

    POR2D_Omega1_min = 600.0
    POR2D_Omega1_max = 1800.0
    POR2D_nOmega1 = 1201

    POR2D_Omega2_min = -1200.0
    POR2D_Omega2_max = 1800.0
    POR2D_nOmega2 = 3001
    
    POR2D_PULSE_CONF = '123'
    POR2D_PULSE_VEC = 'HHH'

    PARAM = {
        'f_energy_dipole': f_energy_dipole,
        'f_disp': f_disp,
        'basetime': basetime,
        'num': num,
        'domega': domega,
        'beta': beta,
        'Gamma': Gamma,
        'Gamma_g': Gamma_g,
        'sigma': sigma,
        'vibquanta_max': vibquanta_max,
        'ABS_OmegaL_min': ABS_OmegaL_min,
        'ABS_OmegaL_max': ABS_OmegaL_max,
        'ABS_nOmegaL': ABS_nOmegaL,       
        'CW_read_alpha': CW_read_alpha,
        'CW_write_alpha': CW_write_alpha,
        'CW_cal_2DCW': CW_cal_2DCW,
        'CW_save_txt': CW_save_txt,
        'CW_save_npy': CW_save_npy,
        'CW_OmegaS_min': CW_OmegaS_min,
        'CW_OmegaS_max': CW_OmegaS_max,
        'CW_nOmegaS': CW_nOmegaS,
        'CW_OmegaL_min': CW_OmegaL_min,
        'CW_OmegaL_max': CW_OmegaL_max,
        'CW_nOmegaL': CW_nOmegaL,
        'PU1d_OmegaS_min': PU1d_OmegaS_min,
        'PU1d_OmegaS_max': PU1d_OmegaS_max,
        'PU1d_nOmegaS': PU1d_nOmegaS,
        'TD_read_alpha': TD_read_alpha,
        'TD_Omega1_min': TD_Omega1_min,
        'TD_Omega1_max': TD_Omega1_max,
        'TD_nOmega1': TD_nOmega1,
        'TD_Omega2_min': TD_Omega2_min,
        'TD_Omega2_max': TD_Omega2_max,
        'TD_nOmega2': TD_nOmega2,
        'TD_configuration': TD_configuration,
        'POR1D_OmegaS_min': POR1D_OmegaS_min,
        'POR1D_OmegaS_max': POR1D_OmegaS_max,
        'POR1D_nOmegaS': POR1D_nOmegaS,
        'POR1D_PULSE_CONF': POR1D_PULSE_CONF,
        'POR1D_PULSE_VEC': POR1D_PULSE_VEC,
        'POR2D_Omega1_min': POR2D_Omega1_min,
        'POR2D_Omega1_max': POR2D_Omega1_max,
        'POR2D_nOmega1': POR2D_nOmega1,
        'POR2D_Omega2_min': POR2D_Omega2_min,
        'POR2D_Omega2_max': POR2D_Omega2_max,
        'POR2D_nOmega2': POR2D_nOmega2,
        'POR2D_PULSE_CONF': POR2D_PULSE_CONF,
        'POR2D_PULSE_VEC': POR2D_PULSE_VEC,

    }
    
    int_values = [
        'vibquanta_max','num',
        'CW_nOmegaS','CW_nOmegaL',
        'ABS_nOmegaL',
        'PU1d_nOmegaS',
        'TD_nOmega1','TD_nOmega2',
        'POR1D_nOmegaS', 'POR2D_nOmega1', 'POR2D_nOmega2'
    ]
    bool_values = [
        'CW_read_alpha','CW_write_alpha','CW_cal_2DCW','CW_save_txt','CW_save_npy',
        'TD_read_alpha'
    ]
    str_values = ['f_energy_dipole', 'f_disp',
        'PU_pulses','TD_configuration', 
        'POR1D_PULSE_CONF', 'POR1D_PULSE_VEC',
        'POR2D_PULSE_CONF', 'POR2D_PULSE_VEC'
    ]
    
    inp_fh = open("main.inp",'r')
    lines = inp_fh.readlines()
    for line in lines:
        line = line.strip()
        if len(line) < 2 : continue
        if line[0] == '#': continue
        #print line
        
        key_name = line.split()[0]
        key_value = line.split()[1]
        if key_name in int_values:
            PARAM[key_name] = int(key_value)
        elif key_name in str_values:
            PARAM[key_name] = key_value   
        elif key_name in bool_values:
            if key_value[0] == 'F' or key_value[0] == 'f':
                PARAM[key_name] = False
            else:
                PARAM[key_name] = True
        else:
            PARAM[key_name] = float(key_value)
    
    PARAM['basetime'] = PARAM['basetime'] / (2*np.pi*2.99792)
    PARAM['num'] = 2 ** PARAM['num']
    
    if PARAM['PU_pulses'] == 'energies':
        SYSTEM = initialize_system(PARAM)
        PARAM['PU_n_pulses'] = len(SYSTEM['energies'])
        PARAM['PU_pulses_list'] = SYSTEM['energies']
    elif PARAM['PU_pulses'] == 'read':
        for line in lines:
            line = line.strip()
            if len(line) < 2: continue
            if line[0] == '#': continue
            
            keyname = line.split()[0]
            if keyname == 'PU_n_pulses':
                PARAM[keyname] = line.split()[1]
            if keyname == 'PU_pulses_list':
                plist = [float(line.split()[1+i]) for i in range(len(line.split())-1)]
        PARAM['PU_pulses_list'] = plist
    else:
        print 'Wrong value for "PU_pulses" in main.inp'
        print 'Only "energies" or "read" is accepted.'
        sys.exit()
    inp_fh.close()
      
    return PARAM

def initialize_system(PARAM):
    eV2wavenum = 8065.541154
    vibfreq_scale = 0.97
    
    # load excitation energies and dipoles, find number of excited states
    f_energy_dipole = PARAM['f_energy_dipole']
    f_disp = PARAM['f_disp']

    energies_dipoles = np.loadtxt(f_energy_dipole)
    
    energies = energies_dipoles[:,0]
    ne = len(energies)
    energies *= eV2wavenum
    dipoles = energies_dipoles[:,-1]
    dipvecs = energies_dipoles[:,1:-1]
    
    # read displacements and vibration frequencies
    disp = np.loadtxt(f_disp)

    vibfreqs = disp[:,0] * vibfreq_scale
    nmodes = len(vibfreqs)

    disps = np.transpose(disp[:,1:])
    
    SYSTEM = {
        'name': 'whatever',
        'energies': energies,
        'dipoles': dipoles,
        'dipvecs': dipvecs,
        'vibfreqs': vibfreqs,
        'disps': disps
    }
    
    return SYSTEM


# Die function
def die(string):
    print string
    sys.exit(0)


# define universal function coth from np.cosh and np.sinh
coth_ufunc = np.frompyfunc(lambda x: np.cosh(x)/np.sinh(x), 1, 1)

# Factorial function
def factorial(n):
    if n == 0: return 1
    if n > 0:
        result = 1
        for i in range(1,n):
            result *= (i+1)
        return result
    else:
        string = "TypeError: factorial needs nonnegative integer as argument."
        die(string)
            


# Franck-Condon integrals
# FCfunc(D,n,m) = < (n)_e | (m)_g >^j
def FCfunc(D, n, m):
    result = 0.0
    for k in range(n+1):
        for l in range(m+1):
            if (n-k) != (m-l): continue
            result += (-1)**l * D**(k+l) / \
                (factorial(k) * factorial(l) * factorial(m-l))
    result *= np.sqrt(factorial(n)*factorial(m))
    result *= np.exp(-D**2/2)
    return result
