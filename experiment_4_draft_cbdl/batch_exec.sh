#!/bin/bash
mapfile -t files < <(ls -v *_batch_*)

for file in "${files[@]}"; do
    
    echo "$file"
    
done
