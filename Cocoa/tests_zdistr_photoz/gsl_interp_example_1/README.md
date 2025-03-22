## Diogo H. F. de Souza -  Wed Mar 19 2025

# - 1st example of https://www.gnu.org/software/gsl/doc/html/interp.html#d-interpolation-example-programs
# - gsl_interp_example_1.c interpolates data (x,y) with cubic spline method
# - to run this example, type in the command line:

## STEP 1: Setup the conda env
conda activate diogo ## or your enviroment
cd cocoa/Cocoa       ## or you CoCoA
source start_cocoa   ## to have access to GSL

## STEP 2: Actual run
gcc -Wall -I/usr/local/include/ -c gsl_interp_example_1.c
gcc gsl_interp_example_1.c -lgsl -lcblas -lm
./a.out > interp.dat

## STEP 3: Plot interp.dat and compare with https://www.gnu.org/software/gsl/doc/html/interp.html#id6
python gsl_interp_example_1.py ## should create a figure named "test.pdf"