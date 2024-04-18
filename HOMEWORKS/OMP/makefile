# # Makefile for compiling and running code with different variable values

# # Compiler
# CC = icc

# # Compiler flags
# CFLAGS = -fopenmp -g

# # Source file
# SRC = omp_homework.c

# # Executable name
# TARGET = omp_homework

# # Variable values
# N = 1000 10000 30000
# num_threads = 1 2 4 6 8 12 16 20 24

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

run:
	@$(foreach n, $(N), \
		$(foreach THREAD, $(THREADS), \
			echo Running with n=$(n) and THREADS=$(THREAD); \
			./$(TARGET) $(n) $(THREAD); \
		) \
	)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

