#!/usr/bin/env python3

import shutil
import os

import numpy as np

from decimal import Decimal

def decimal_range(start, stop, step):
    
    start = Decimal(str(start))
    stop = Decimal(str(stop))
    step = Decimal(str(step))
    
    current = start
    while current < stop:
        yield current
        current += step


original_file = "py_ex_3_gpuL.slurm.epochs_udist_template"

##  Uniform distribution loop.

for i in list(decimal_range('-1.5', '1.5', '0.25')):

    udist = str(i)

    ##  Training epochs loop.
    
    for epochs in range(5, 5*5+1, 5):

        new_filename = f"{original_file}.epochs_{epochs}_udist_{udist}"
    
        shutil.copy(original_file, new_filename)

        with open(new_filename, 'r') as f:

            content = f.read()

        edited_content = content.replace("EPOCHS", str(epochs)).replace("UDIST", udist)

        with open(new_filename, 'w') as f:

            f.write(edited_content)

