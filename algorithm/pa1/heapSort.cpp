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

void MaxHeapify(word_module** A, int root , int total_word_count){ 
	//cout<<"MaxHeapify "<<root<<endl;
    int left = 2 * root;     
    int right = 2 * root + 1;     
    int largest = root;                 

    if(left <= total_word_count){
    	if(A[left]->key > A[largest]->key){
    		largest = left;
    	}
    }

    if(right <= total_word_count){
    	if(A[right]->key > A[largest]->key){
    		largest = right;
    	}
    } 
         
    if (largest != root) {                      
        //swap(A[largest], A[root]);
        word_module* tmp = A[largest];
        A[largest] = A[root];
        A[root] = tmp;          
        MaxHeapify(A, largest, total_word_count);      
    }
}

void BuildMaxHeap(word_module** A , int total_word_count){ 
    for (int i = (int)total_word_count/2; i >= 1 ; i--) {
    	//cout<<"BuildMaxHeap "<<i<<endl;
        MaxHeapify( A , i , total_word_count);     
    }
}

void HeapSort(word_module** A , int total_word_count){
 	//cout<<"HeapSort "<<endl;
    BuildMaxHeap(A , total_word_count);                              

    int size = total_word_count;                     
    for (int i = total_word_count ; i >= 2; i--) {
    	//cout<<"HeapSort "<<i<<endl;
        swap(A[1], A[i]);                       
        size--;
        MaxHeapify(A, 1, size);                     
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
	
	// Heap sort
	HeapSort(arr , total_word_count); 

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