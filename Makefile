UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
CXX = clang++
endif

ifeq ($(UNAME), Linux)
CXX = g++
endif

extract_data: extract_data.cxx 
	$(CXX) -o extract_data extract_data.cxx -lz `root-config --cflags --libs`
clean:
	rm extract_data

