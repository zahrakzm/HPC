# Makefile for compiling and running code with different variable values

# Compiler
CC = icc

# Compiler flags
CFLAGS = -Wall -Wextra

# Source file
SRC = omp_homework.c

# Executable name
EXEC = your_executable

# Variable values
VAR1_VALUES = value1 value2 value3
VAR2_VALUES = valueA valueB valueC

# Define all combinations of variable values
ALL_COMBINATIONS := $(foreach v1,$(VAR1_VALUES),$(foreach v2,$(VAR2_VALUES),$(v1)_$(v2)))

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
