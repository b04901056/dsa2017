#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include "hmm.h"
#define DBL_MIN         2.2250738585072014e-308
using namespace std;
   
HMM *hmm; 
vector<string> buffer;
int **buffer_to_int;
vector<int> acc; 
long double **delta; 
 
void convert_to_int( vector<string> buffer , int **buffer_to_int ){  
	for(int i=0;i<buffer.size();i++){
    	for(int j=0;j<50;j++){
    		 if(buffer[i][j]=='A') buffer_to_int[i][j] = 0;
    		 	else if(buffer[i][j]=='B') buffer_to_int[i][j] = 1;
    		 		else if(buffer[i][j]=='C') buffer_to_int[i][j] = 2;
    		 			else if(buffer[i][j]=='D') buffer_to_int[i][j] = 3;
    		 				else if(buffer[i][j]=='E') buffer_to_int[i][j] = 4;
    		 					else buffer_to_int[i][j] = 5 ;
    	}
    } 
}  
 
long double get_max_delta( long double** delta , HMM hmm , int* obs ){
	for(int i=0;i<6;i++) delta[i][0] = hmm.initial[i] * hmm.observation[obs[0]][i];
		//for(int i=0;i<6;i++) cout<<delta[i][0]<<endl ;
	for(int t=1;t<50;t++){
		for(int j=0;j<6;j++){
			long double max = 0; 
			for(int i=0;i<6;i++){ 
				if( delta[i][t-1] * hmm.transition[i][j] > max ) max = delta[i][t-1] * hmm.transition[i][j];
			}		
			delta[j][t] = max * hmm.observation[obs[t]][j];
		}
	}	
	long double result = 0; 	
	for(int i=0;i<6;i++) {
		//cout<<"max= "<<delta[i][49]<<endl;
		if( delta[i][49] > result ) result = delta[i][49];
	}
	return result; 
}
// ./test  modellist.txt  testing_data.txt  result.txt
int main(int argc, const char* argv[])
{ 
	delta = new long double*[6];
	for(int i=0;i<6;i++) delta[i] = new long double[50];
	buffer_to_int = new int*[2500];
	for(int i=0;i<2500;i++) buffer_to_int[i] = new int[50]; 
	hmm = new HMM[5];
	load_models( argv[1] , hmm , 5 );


	ifstream fin(argv[2]);  
    string s;   
    while( getline(fin,s) ) buffer.push_back(s);  
    convert_to_int( buffer , buffer_to_int ); 

    ifstream fin_(argv[3]);    
    while( getline(fin_,s) ) acc.push_back(s[7]-'0'); 
 
 	float count = 0;
 	vector<int> rec;
 	vector<long double> pb;

    for(int k=0;k<buffer.size();k++){ 
    	int record = -1 ;
    	long double max= 0 ;
    	for(int i=0;i<5;i++){
    		long double a = get_max_delta( delta , hmm[i] , buffer_to_int[k] );
    		if( a > max){
    			record = i+1;
    			max = a;
    		}
    	}

    	cout<<"========================================"<<endl;
    	cout<<"model: "<<record<<"    "<<"prob= "<<max<<" "<<acc[k]<<endl;
    	if(record==acc[k]) count++;
    	rec.push_back(record);
    	pb.push_back(max);
    }  
    count /= 2500 ;
    cout<<"accuracy: "<< count <<endl;


    fstream file;
    file.open(argv[4], ios::out);
    for(int i = 0; i < 2500; i++) {
		file << "model0"<< rec[i] << ".txt " << pb[i] << "\n";
	}  

	fstream f;
    f.open(argv[5], ios::out);
	f << count << "\n";  


    for(int i=0;i<6;i++) delete delta[i]; 
	delete delta;
	for(int i=0;i<2500;i++) delete buffer_to_int[i]; 
	delete buffer_to_int;
	delete hmm; 
}