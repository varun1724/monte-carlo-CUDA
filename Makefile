# Makefile for MonteCarloAmerican project

# Compiler
NVCC = nvcc

# Target executable name
TARGET = MonteCarloAmerican

# Find all source files in src/ (both .cpp and .cu)
SRC_CPP = $(wildcard src/*.cpp)
SRC_CU  = $(wildcard src/*.cu)
SRC     = $(SRC_CPP) $(SRC_CU)

# Compiler flags (adjust the architecture as needed)
NVCCFLAGS = -O3 -arch=sm_75 -x cu -std=c++11

# Libraries to link
LIBS = -lcublas

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) $(LIBS) -o $(TARGET)

# Clean up build artifacts
clean:
	rm -f $(TARGET)

.PHONY: all clean
