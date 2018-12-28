SRIPATH=

MACHINE_TYPE=

LM =  

CXX=g++
CXXFLAGS=-O3 -I$(SRIPATH)/include

vpath lib%.a $(SRIPATH)/lib/$(MACHINE_TYPE)

TARGET=mydisambig
SRC = mydisambig.cpp
OBJ = $(SRC:.cpp=.o) 

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ) -loolm -ldstruct -lmisc
	$(CXX) $(LDFLAGS) -o $@ $^ 

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

map: transform.py
	python3 transform.py  

run: $(OBJ) -loolm -ldstruct -lmisc
	$(CXX) $(LDFLAGS) -o $@ $^
	./run $(LM)  ZhuYin-Big5.map 1.txt result2/1.txt
	./run $(LM)  ZhuYin-Big5.map 2.txt result2/2.txt 
	./run $(LM)  ZhuYin-Big5.map 3.txt result2/3.txt 
	./run $(LM)  ZhuYin-Big5.map 4.txt result2/4.txt 
	./run $(LM)  ZhuYin-Big5.map 5.txt result2/5.txt 
	./run $(LM)  ZhuYin-Big5.map 6.txt result2/6.txt 
	./run $(LM)  ZhuYin-Big5.map 7.txt result2/7.txt 
	./run $(LM)  ZhuYin-Big5.map 8.txt result2/8.txt 
	./run $(LM)  ZhuYin-Big5.map 9.txt result2/9.txt 
	./run $(LM)  ZhuYin-Big5.map 10.txt result2/10.txt  
	

clean:
	$(RM) $(OBJ) $(TARGET)

