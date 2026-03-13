#!/usr/bin/env python3
import shutil
import os

import random

original_file = "py_ex_5_gpuL_test.slurm.batch_01"

for i in range(10, 17):

    batch_n = str(2**i)

    new_filename = original_file.replace("01", batch_n)

    shutil.copy(original_file, new_filename)

    with open(original_file, 'r') as f:

        content = f.read()

    edited_content = content.replace("BATCH_N", batch_n)

    with open(new_filename, 'w') as f:

        f.write(edited_content)

