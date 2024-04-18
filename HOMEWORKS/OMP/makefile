# # Makefile for compiling and running code with different variable values

# # Compiler
# CC = icc

# # Compiler flags
# CFLAGS = -fopenmp -g

# # Source file
# SRC = omp_homework.c

# # Executable name
# EXEC = omp_homework

# # Variable values
# N = 1000 10000 30000
# num_threads = 1 2 4 6 8 12 16 20 24

# # Define target to run the code with different variable values
# all: $(foreach v1,$(Var1_values),$(foreach v2,$(Var2_values),run_$(v1)_$(v2)))

# # Rule to compile and link executable with specific variable values
# run_%:
#     @var1=$(word 1,$(subst _, ,$*)); \
#     var2=$(word 2,$(subst _, ,$*)); \
#     echo "Running with Var1 = $$var1, Var2 = $$var2"; \
#     $(CC) $(CFLAGS) -DVar1=$$var1 -DVar2=$$var2 $(SRC) -o $(EXEC)_$$var1_$$var2; \
#     ./$(EXEC)_$$var1_$$var2

# # Clean rule
# clean:
#         rm -f $(EXEC)_*

# Define the C compiler
CC = icc

# Define C compiler flags (feel free to modify)
CFLAGS = -Wall

# Target to build the final executable (all)
all: main

# Target to compile the source file into an object file (main.o)
main.o: main.c
    $(CC) $(CFLAGS) -c omp_homework.c -o main.o

# Target to clean up generated files (clean)
clean:
    rm -f *.o

# By default, run 'make all'
.PHONY: all clean

