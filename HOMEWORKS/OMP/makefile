# Makefile for compiling and running code with different variable values

# Compiler
CC = icc

# Compiler flags
CFLAGS = -g -fopenmp

# Source file
SRC = omp_homework.c

# Executable name
EXEC = omp_homework

# Variable values
SIZE = 1000 10000 30000
NUM_THREADS = 1 2 4 8 12 16 20 24 30

# Define all combinations of variable values
ALL_COMBINATIONS := $(foreach v1,$(SIZE),$(foreach v2,$(NUM_THREADS),$(v1)_$(v2)))

# Define targets for each combination of variable values
all: $(addprefix run_,$(ALL_COMBINATIONS))

# Rule to compile and link executable with specific variable values
$(EXEC)_%: $(SRC)
    $(CC) $(CFLAGS) -DVAR1=$(word 1,$(subst _, ,$*)) -DVAR2=$(word 2,$(subst _, ,$*)) $(SRC) -o $@

# Rule to run the executable
run_%: $(EXEC)_% 
    ./$< $(word 1,$(subst _, ,$*)) $(word 2,$(subst _, ,$*))

# Clean rule
clean:
    rm -f $(EXEC)_* $(EXEC)
