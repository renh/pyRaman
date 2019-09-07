#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==============================================================================
# Module documentation
"""
FileName	: xyz2rtp.py
Purpose		: convert cartesian coordinates to spherical
Author		: Hao Ren
Version		: 0.1
Date		: , 2011
"""
#==============================================================================

#==============================================================================
# Module imports
import math
#==============================================================================

dip_f = 'energies_dipoles.inp'
fh = open(dip_f,'r')

dump = fh.readlines()
fh.close()

dipoles = []
for l in dump:
    if l.startswith('#'): continue
    dipoles.append(
        [float(i) for i in l.split()][1:4]
    )

nex = len(dipoles)
for i in range(nex):
    x,y,z = dipoles[i]
    r = math.sqrt(x*x+y*y+z*z)
    t = math.acos(z/r)/math.pi
    p = math.atan(y/x)/math.pi
    if p < 0: p += 2
    print r, t, p
