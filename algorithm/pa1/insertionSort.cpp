#include <iostream>
#include <cstdio> 
#include <string>
#include "parser.h"
#include <fstream>
using namespace std;

class word_module{
public:
	word_module(int i , string a){
		key = a;
		original_position = i;
	}
	string key;
	int original_position;
};

int main( int argc, char** argv ){ 
	word_module** arr;
	AlgParser p;
	AlgTimer t; 

	p.Parse( argv[1] ); 
	int total_word_count = p.QueryTotalStringCount(); 
	arr = new word_module*[total_word_count+1]; 

	// Start timer
	t.Begin(); 
 
 	arr[0] = new word_module(0 , "first_word!!!"); 
	for( int i = 1 ; i <= total_word_count ; i++ ){ 
		arr[i] = new word_module(i , p.QueryString(i-1)); 
	} 

	// Insertion sort
	
	for(int j = 2 ; j <= total_word_count ; j++ ){ 
		word_module* tmp = arr[j];
		int i = j - 1;
		while( (i > 0) && (arr[i]->key > tmp->key) ){
			arr[i+1] = arr[i];
			i--; 
		}
		arr[i+1] = tmp;
	}  

	// Display the accumulated time
	cout << "The execution spends " << t.End() << " seconds" << endl;

	// Output 
	ofstream filePtr;                       
	filePtr.open(argv[2], ios::out); 
	filePtr << total_word_count <<endl;               
	for(int i = 1 ; i <= total_word_count ; i++ ){
		filePtr << arr[i]->key << " " << arr[i]->original_position << endl ;
	}         
	filePtr.close();                       


	// Delete dynamic memory
	for( int i = 0 ; i < total_word_count ; i++ ){ 
		delete arr[i]; 
	} 
	delete arr; 

	return 0;
}