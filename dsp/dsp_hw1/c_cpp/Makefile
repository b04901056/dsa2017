.PHONY: all clean

CFLAGS+= -g -std=c++11 
LDFLAGS+=-lm     # link to math library

TARGET=train test 

all: $(TARGET)

train: train.cpp
	g++ $(CFLAGS) -o $@ $<
	# type make/make all to compile test_hmm
test: test.cpp
	g++ $(CFLAGS) -o $@ $<
clean:
	$(RM) $(TARGET)   # type make clean to remove the compiled file
