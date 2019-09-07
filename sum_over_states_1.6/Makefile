FC=ifort
EXE=main
FFLAGS= -O2 #-assume byterecl -O2
#MKL_INT=-L /opt/intel/mkl/lib/intel64 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread
#LAPACK = -L /opt/intel/mkl/lib/intel64 -lmkl_lapack95_lp64 
#LAPACK = -L/usr/lib64 -llapack

LIBS = $(MKL_INT) $(LAPACK)

.SUFFIXES:.f90

.f90.o:
	$(FC) $(FFLAGS)  -c $< 

%.o:%.mod

objs=main.o prec.o io.o fdf.o

$(EXE):$(objs)
	$(FC) $(FFLAGS)  -o $(EXE) $(objs) $(LIBS)

main.o:prec.o io.o fdf.o
io.o:prec.o

clean:
	rm -f *.o *.mod $(EXE)
