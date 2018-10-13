#include <iostream>
#include <cstdio> 
#include <string>
#include "parser.h"
#include <fstream>
using namespace std;
string word_max = "zzzzzzzzzzzzzzzzzzzzzzzzz";
 
class word_module{
public:
	word_module(int i , string a){
		key = a;
		original_position = i;
	}
	string key;
	int original_position;
};

void Merge(word_module** A , int p , int q , int r){
	//cout<<"Merge: "<<" p= "<<p<<" q= "<<q<<" r= "<<r<<endl;
	int n1 = q - p + 1 ;
	int n2 = r - q ;
	int i = 1;
	int j = 1;
	word_module** tmp_a;
	word_module** tmp_b;
	tmp_a = new word_module*[n1+2];
	tmp_b = new word_module*[n2+2];
	for(int s=1;s<=n1;s++){
		tmp_a[s] = A[p + s - 1] ;
	}
	for(int s=1;s<=n2;s++){
		tmp_b[s] = A[q + s] ;
	}
	tmp_a[n1+1] = new word_module( 0 , word_max );
	tmp_b[n2+1] = new word_module( 0 , word_max );

	for(int k=p;k<=r;k++){
		if( tmp_a[i]->key <= tmp_b[j]->key ){
			A[k] = tmp_a[i];
			i++;
		}
		else{
			A[k] = tmp_b[j];
			j++;
		}
	}
	delete tmp_a[n1+1];
	delete tmp_b[n2+1];
	delete tmp_a;
	delete tmp_b;
}

void MergeSort(word_module** A, int p, int r){
	//cout<<"MergeSort: "<<" p= "<<p<<" r= "<<r<<endl;
	if( p < r ){
		int q = ( p + r ) / 2 ;
		MergeSort ( A , p , q) ;
		MergeSort ( A , q + 1 , r) ;  
		Merge ( A , p , q , r) ; 
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

	// Merge sort  
	MergeSort(arr,1,total_word_count); 

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