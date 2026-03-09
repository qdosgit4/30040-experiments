#!/bin/bash

for file in *"_batch_"*; do [ -e "$file" ] && echo "Processing $file"; done
