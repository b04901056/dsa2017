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

void swap(word_module* a , word_module* b){
	string key_tmp = a->key;
	int original_position_tmp = a->original_position;
	a->key = b->key;
	a->original_position = b->original_position;
	b->key = key_tmp;
	b->original_position = original_position_tmp;
}

int Partition(word_module** A, int p, int r){
	//cout<<"Partition "<<p<<" "<<r<<endl;
	word_module* x = A[p];
    int i = p - 1 ;
    int j = r + 1 ;
    while(true){
    	j--;
    	while(A[j]->key > x->key){
    		j--;
    		//cout<<"j= "<<j<<endl;
    	}
    	i++; 
    	while(A[i]->key < x->key){
    		i++; 
    		//cout<<"i= "<<i<<endl;
    	}  
		if(i < j){
			//cout<<"swap "<<i<<" "<<j<<endl;
			//swap(A[i] , A[j]);
			word_module* tmp = A[i];
        	A[i] = A[j];
        	A[j] = tmp; 
		} 
		else return j;
    }
}

void QuickSort(word_module** A, int p, int r){
	//cout<<"QuickSort "<<p<<" "<<r<<endl;
    if (p < r) {
        int q = Partition(A, p, r);
        QuickSort(A, p, q);
        QuickSort(A, q + 1, r);
    }
}

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

	// Quick sort
	QuickSort(arr , 1 , total_word_count); 

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