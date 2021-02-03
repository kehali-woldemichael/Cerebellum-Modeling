PROGRAM = main
SRCS_CU  = $(wildcard *.cu */*.cu)
OBJS_CU  = ${SRCS_CU:.cu=.o}
CC = @nvcc
DEBUG = -g -G -DTHRUST_DEBUG
OPTIMISATIONFLAGS = -use_fast_math -Xcompiler "-w -O3 -ffast-math -fno-finite-math-only -march=native"
NVCCFLAGS = -w  -I. -std=c++11
GPU_ARCH_FLAGS =  -arch=sm_61
LFLAGS = -lcurand

# Add optimisation flags
all: NVCCFLAGS += $(OPTIMISATIONFLAGS)
all: executable

# Add debug flags
debug: GPU_ARCH_FLAGS += $(DEBUG)
debug: executable

.PHONY: all debug executable clean

executable: $(OBJS_CPP) $(OBJS_CU)
	$(CC) $(GPU_ARCH_FLAGS) $(LFLAGS) $(OBJS_CU) -o $(PROGRAM)

clean:
	rm -f $(OBJS_CPP) $(OBJS_CU) $(PROGRAM) results/*

%.o : %.cu
	$(CC) $(GPU_ARCH_FLAGS) $(NVCCFLAGS) -dc $< -o $@
