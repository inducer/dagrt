#! /bin/bash
set -e
rm -Rf stability-tables
mkdir stability-tables
python `which runalyzer` -m direct-data/lores.dat make-tables.py
