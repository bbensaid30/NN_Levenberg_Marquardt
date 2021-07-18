CC=g++
CXXFLAGS = -pedantic -Wall -Wextra -fexceptions -std=c++17 -g -mfma -fopenmp -I /usr/include/eigen3 -I /usr/include/python3.8/ -I /usr/local/lib/python3.8/dist-packages/numpy/core/include/ -I /home/bensaid/Documents/shaman/PREFIX/include/ -I /home/bensaid/Documents/EigenRand-0.3.5/ -D SHAMAN_UNSTABLE_BRANCH -D SHAMAN_TAGGED_ERROR
LDFLAGS = -lpython3.8 -lshaman
LDLIBS = -L /home/bensaid/Documents/shaman/PREFIX/lib
EXEC = forward
SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) -o $@ $^ $(CXXFLAGS) $(LDLIBS) $(LDFLAGS)
	 
main.o: init.h test.h data.h study_base.h utilities.h study_graphics.h
propagation.o: activations.h
training.o: propagation.h utilities.h scaling.h
training_entropie.o: propagation.h utilities.h scaling.h
test.o: init.h training.h training_entropie.h utilities.h
addStrategy.o: propagation.h utilities.h
study_base.o: init.h propagation.h training.h training_entropie.h utilities.h addStrategy.h
study_graphics.o: study_base.h
 
%.o: %.c
	$(CC) -o $@ -c $< $(CXXFLAGS) $(LDLIBS) $(LDFLAGS)

clean:
	rm -f *.o core

mrproper: clean
	rm -f $(EXEC)
