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

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

run:
	@$(foreach N, $(SIZE), \
		$(foreach THREAD, $(NUM_THREADS), \
			echo Running with N=$(N) and NUM_THREADS=$(THREAD); \
			./$(TARGET) $(N) $(THREAD); \
		) \
	)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
