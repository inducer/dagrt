#! /bin/sh
rm -Rf contour-plots
mkdir contour-plots
#python make-contour-plots.py direct-data/hires-mrab.dat
python make-contour-plots.py direct-data/hires.dat
