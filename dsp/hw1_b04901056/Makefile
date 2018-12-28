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

execute : train
	./train 100 model_init.txt seq_model_01.txt model_01.txt;\
	./train 100 model_init.txt seq_model_02.txt model_02.txt;\
	./train 100 model_init.txt seq_model_03.txt model_03.txt;\
	./train 100 model_init.txt seq_model_04.txt model_04.txt;\
	./train 100 model_init.txt seq_model_05.txt model_05.txt
predict : test
	./test modellist.txt  testing_data1.txt testing_answer.txt result1.txt acc.txt
	./test modellist.txt  testing_data2.txt testing_answer.txt result2.txt acc2.txt
	rm acc2.txt

		
