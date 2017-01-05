CXXFLAGS=-O3 -std=c++11 -pedantic -Wall

all: obf
	cat README.md | ./$^ -t

debug: obf.cc
	$(CXX) $(CXXFLAGS) -g -Wl,--no-as-needed -lSegFault -fno-inline -o obf obf.cc
