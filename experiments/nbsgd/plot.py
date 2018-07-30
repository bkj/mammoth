#!/usr/bin/env python

"""
    plot.py
"""

import os
import sys
import json
import numpy as np

from rsub import *
from matplotlib import pyplot as plt


for p in sys.argv[1:]:
    x = list(map(json.loads, open(p).read().splitlines()))
    # val_acc  = np.array([xx['val_acc'] for xx in x])
    test_acc = np.array([xx['test_acc'] for xx in x])
    
    # _ = plt.plot(val_acc, alpha=0.75, label='%s_valid' % os.path.basename(p))#, c='red')
    _ = plt.plot(test_acc, alpha=0.75, label='%s_test' % os.path.basename(p))#, c='blue')

_ = plt.legend(fontsize=6, loc='lower left')
show_plot()