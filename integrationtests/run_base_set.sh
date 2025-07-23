#!/bin/bash

bsub -q interactive -M 24G -c 2 -R "rusage[mem=8G]" -gpu "num=1:mode=shared:j_exclusive=yes" 'cd /dccstor/terratorch/users/rkie/gitco/terratorch && source .venv/bin/activate && cd integrationtests && time pytest test_base_set.py'

# For jbsub 
#jbsub -q x86_1h -mem 24G -cores 1x2+1 -interactive 'cd /dccstor/terratorch/users/rkie/gitco/terratorch && source .venv/bin/activate && cd integrationtests && time pytest test_base_set.py'

#/opt/share/exec/jbsub8 -q x86_1h -mem 24G -cores 1x2+1 -interactive cd /dccstor/terratorch/users/rkie/gitco/terratorch && source .venv/bin/activate && cd integrationtests && time pytest test_base_set.py
