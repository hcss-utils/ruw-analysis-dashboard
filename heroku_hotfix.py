#!/usr/bin/env python
# coding: utf-8

"""
Heroku-specific optimizations to reduce memory usage
"""

import os
import gc
import logging

def optimize_for_heroku():
    """Apply Heroku-specific optimizations"""
    
    # Force garbage collection
    gc.collect()
    
    # Reduce pandas memory usage
    import pandas as pd
    pd.options.mode.chained_assignment = None
    
    # Set smaller cache sizes
    os.environ['CACHE_TIMEOUT'] = '300'  # 5 minutes instead of 1 hour
    
    # Limit worker threads
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    logging.info("Applied Heroku optimizations")

# Call this at startup
optimize_for_heroku()