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
#==============================================================================

try:
    fh = open("onp.fchk", "r")
except:
    print "open error, exit"
    raise SystemExit
