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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys
#==============================================================================

try:
    script, iex, im = sys.argv
except:
    print "Usage: %s iex im" % sys.argv[0]
    raise SystemExit

# rcParams
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.minor.size'] = 3.5
mpl.rcParams['xtick.major.width'] = .8
mpl.rcParams['xtick.minor.width'] = .6
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.minor.size'] = 3.5
mpl.rcParams['ytick.major.width'] = .8
mpl.rcParams['ytick.minor.width'] = .6

data = np.load('ang-%s-%s.npy' % (iex,im))
data = np.abs(data)
vm = np.max(data)

xtick_pos = np.arange(7)*12
ytick_pos = np.arange(7)*6
ytick_lab = ["0", 
             r'$\frac{1}{6}\pi$',
             r'$\frac{1}{3}\pi$',
             r'$\frac{1}{2}\pi$',
             r'$\frac{2}{3}\pi$',
             r'$\frac{5}{6}\pi$',
             r'$\pi$'
            ]
xtick_lab = ["0", 
             r'$\frac{1}{3}\pi$',
             r'$\frac{2}{3}\pi$',
             r'$\pi$',
             r'$\frac{4}{3}\pi$',
             r'$\frac{5}{3}\pi$',
             r'$2\pi$'
            ]
mlx = MultipleLocator(1)
mly = MultipleLocator(1)

contour_levels = np.concatenate(
    (np.linspace(0,.3*vm,15),np.linspace(.3*vm,.7*vm,8),np.linspace(.7*vm,vm,15))
)

fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(111,aspect='equal')
ax1.set_xticks(xtick_pos)
ax1.set_xticklabels(xtick_lab)
ax1.set_yticks(ytick_pos)
ax1.set_yticklabels(ytick_lab)
ax1.xaxis.set_minor_locator(mlx)
ax1.yaxis.set_minor_locator(mly)
ax1.contour(data,origin='lower',
           levels = contour_levels
           )
ax1.set_xlabel(r'$\phi$')
ax1.set_ylabel(r'$\theta$')
plt.savefig('ang-%s-%s.pdf'%(iex,im))

