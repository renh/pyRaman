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
#==============================================================================

def proj(u,v):
    return np.dot(v,u) / np.dot(u,u) * u

def mode(u):
    return np.sqrt(np.dot(u,u))

def gram(V):
    M,N = V.shape
    U = np.zeros([M,N])
    E = np.zeros([M,N])
    U[0] = V[0]
    E[0] = U[0] / mode(U[0])
    for k in range(1,M):
        tmp = np.zeros(N)
        for j in range(k):
            tmp += proj(U[j],V[k])
        U[k] = V[k] - tmp
        E[k] = U[k] / mode(U[k])
    return E

        
