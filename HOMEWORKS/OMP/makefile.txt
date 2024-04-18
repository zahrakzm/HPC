# Makefile for compiling and running code with different variable values

# Compiler
CC = icc

# Compiler flags
CFLAGS = -fopenmp -g

# Source file
SRC = omp_homework.c

# Executable name
EXEC = omp_homework

# Variable values
N = 1000 10000 30000
num_threads = 1 2 4 6 8 12 16 20 24

all:
    @for var1 in $(N); do \
        for var2 in $(num_threads); do \
            echo "Running with Var1 = $$var1, Var2 = $$var2"; \
            $(CC) $(CFLAGS) -DVar1=$$var1 -DVar2=$$var2 $(SRC) -o $(EXEC)_$$var1_$$var2; \
            ./$(EXEC)_$$var1_$$var2; \
        done \
    done

# Clean rule
clean:
    rm -f $(EXEC)_* $(EXEC)
