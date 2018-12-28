#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include "hmm.h"
using namespace std;

int iteration;  
long double **alpha; 
long double **beta; 
long double **gama; 
long double ***ems; 
vector<string> buffer;
int **buffer_to_int;
HMM hmm;
long double ***sigma_ems ;
long double **sigma_gama ; 
long double *sigma_gama_k ; 

void convert_to_int(vector<string> buffer,int **buffer_to_int){  
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

void count_alpha(long double** alpha , HMM hmm , int* obs){ 
	for(int i=0;i<6;i++) alpha[i][0] = hmm.initial[i] * hmm.observation[obs[0]][i];
	for(int t=0;t<49;t++){
		for(int j=0;j<6;j++){
			long double sum=0;
			for(int i=0;i<6;i++) sum += alpha[i][t] * hmm.transition[i][j];
			alpha[j][t+1] = sum * hmm.observation[obs[t+1]][j];		
		}	
	} 
}
void count_beta(long double** beta , HMM hmm , int* obs){
	 for(int i=0;i<6;i++) beta[i][49] = 1 ;
	 for(int t=48;t>=0;t--){
	 	for(int i=0;i<6;i++){
	 		long double sum = 0;
	 		for(int j=0;j<6;j++) sum+= 	hmm.transition[i][j] * hmm.observation[obs[t+1]][j] * beta[j][t+1];
	 		beta[i][t]=sum;
	 	}

	 } 
} 
void count_gama(long double** alpha , long double** beta , long double** gama ){
		for(int t=0;t<50;t++){
			for(int i=0;i<6;i++){
				long double sum = 0;
				for (int j=0;j<6;j++) sum+= alpha[j][t] * beta[j][t];
				gama[i][t] = alpha[i][t] * beta[i][t] / sum;
			}
		}	 	 
} 
void count_ems(long double** alpha , long double** beta , long double*** ems , HMM hmm , int* obs){
		for(int t=0;t<49;t++){
			for(int i=0;i<6;i++){
				for(int j=0;j<6;j++){
					long double sum = 0;
					for(int p=0;p<6;p++){
						for(int q=0;q<6;q++){
							sum += alpha[p][t] * beta[q][t+1] * hmm.transition[p][q] * hmm.observation[obs[t+1]][q];
						}
					}
					ems[t][i][j] = alpha[i][t] * beta[j][t+1] * hmm.transition[i][j] * hmm.observation[obs[t+1]][j] / sum;
				}
			}
		}	  
}  
void update_parameter( long double** sigma_gama , long double*** sigma_ems , long double* sigma_gama_k , long double** gama , long double*** ems , int* obs , HMM* hmm){ 
	for(int i=0;i<6;i++){
		for(int j=0;j<50;j++){
			 sigma_gama[i][j] += gama[i][j];
		}
	}
	for(int t=0;t<49;t++){
		for(int i=0;i<6;i++){
			for(int j=0;j<6;j++){
			 	sigma_ems[t][i][j] += ems[t][i][j]; 
			}
		}
	}

	for(int t=0;t<50;t++){
		 if(obs[t]==0) sigma_gama_k[0]+= gama[0][t];
		 	else if(obs[t]==1) sigma_gama_k[1]+= gama[1][t];
		 		else if(obs[t]==2) sigma_gama_k[2]+= gama[2][t];
		 			else if(obs[t]==3) sigma_gama_k[3]+= gama[3][t];
		 				else if(obs[t]==4) sigma_gama_k[4]+= gama[4][t];
		 					else  sigma_gama_k[5]+= gama[5][t];
	}

}

void update( long double** sigma_gama , long double*** sigma_ems , long double* sigma_gama_k , HMM* hmm){
	for(int i=0;i<6;i++){
		hmm->initial[i] = sigma_gama[i][0] / 10000;
	}
 
	for(int i=0;i<6;i++){
		for(int j=0;j<6;j++){
			float numer = 0;
			float deno = 0;	
			for(int t=0;t<49;t++) numer += sigma_ems[t][i][j];
			for(int t=0;t<50;t++) deno += sigma_gama[i][t];
			hmm->transition[i][j] = numer / deno;
		}
	}

	for(int k=0;k<6;k++){
		for(int i=0;i<6;i++){
			float numer = sigma_gama_k[k];
			float deno = 0;	 
			for(int t=0;t<50;t++) deno += sigma_gama[i][t];
			hmm->observation[k][i] = numer / deno;
		}
	}
	for(int i=0;i<6;i++){
		for(int j=0;j<50;j++){
			 sigma_gama[i][j] = 0 ; 
		}
	}
	for(int t=0;t<49;t++){
		for(int i=0;i<6;i++){
			for(int j=0;j<6;j++){
			 	sigma_ems[t][i][j] = 0 ; 
			}
		}
	}
	for(int i=0;i<6;i++) sigma_gama_k[i] = 0;
}
 
// ./train  iteration  model_init.txt  seq_model_01.txt model_01.txt
int main(int argc, const char* argv[])
{ 
	alpha = new long double*[6];
	for(int i=0;i<6;i++) alpha[i] = new long double[50];  

	beta = new long double*[6];
	for(int i=0;i<6;i++) beta[i] = new long double[50]; 

	gama = new long double*[6];
	for(int i=0;i<6;i++) gama[i] = new long double[50]; 

	buffer_to_int = new int*[10000];
	for(int i=0;i<10000;i++) buffer_to_int[i] = new int[50]; 

	ems = new long double**[49];
	for(int i=0;i<49;i++){
		ems[i] = new long double*[6]; 
		for(int j=0;j<6;j++) ems[i][j] = new long double[6];
	} 

	sigma_ems = new long double**[49];
	for(int i=0;i<49;i++){
		sigma_ems[i] = new long double*[6]; 
		for(int j=0;j<6;j++) sigma_ems[i][j] = new long double[6];
	} 
	sigma_gama = new long double*[6];
	for(int i=0;i<6;i++) sigma_gama[i] = new long double[50]; 

	sigma_gama_k = new long double[6];

  
////////////////////////////////////////////////////////////////////////////////////////////////////////
	iteration = atoi(argv[1]);  
	loadHMM(&hmm,argv[2]); 

	ifstream fin(argv[3]);  
    string s;  
    while( getline(fin,s) ) buffer.push_back(s); 

    convert_to_int( buffer , buffer_to_int ); 

    for(int i=0;i<6;i++){
		for(int j=0;j<50;j++){
			 alpha[i][j] = 0 ; 
		}
	}

	for(int i=0;i<6;i++){
		for(int j=0;j<50;j++){
			 beta[i][j] = 0 ; 
		}
	}

	for(int i=0;i<6;i++){
		for(int j=0;j<50;j++){
			 gama[i][j] = 0 ; 
		}
	}

	for(int t=0;t<49;t++){
		for(int i=0;i<6;i++){
			for(int j=0;j<6;j++){
			 	ems[t][i][j] = 0 ; 
			}
		}
	}

    for(int i=0;i<6;i++){
		for(int j=0;j<50;j++){
			 sigma_gama[i][j] = 0 ; 
		}
	}
	for(int t=0;t<49;t++){
		for(int i=0;i<6;i++){
			for(int j=0;j<6;j++){
			 	sigma_ems[t][i][j] = 0 ; 
			}
		}
	}
	for(int i=0;i<6;i++) sigma_gama_k[i] = 0;

    for(int q=0;q<iteration;q++){
    	cout<<argv[3]<<" : ";
    	cout<<"iteration: "<<q<<endl;
     	for(int i=0;i<buffer.size();i++){ 
    		//cout<<i<<endl;
    		count_alpha( alpha , hmm , buffer_to_int[i] );
    		count_beta( beta , hmm , buffer_to_int[i] );
 			count_gama( alpha , beta , gama );
 			count_ems( alpha , beta , ems , hmm , buffer_to_int[i] ); 
 			update_parameter( sigma_gama , sigma_ems , sigma_gama_k , gama , ems , buffer_to_int[i] , &hmm ); 
    	}
    	update( sigma_gama , sigma_ems , sigma_gama_k , &hmm); 
    }
	FILE * out;
	out = fopen(argv[4], "w");
	dumpHMM( out , &hmm );
	fclose(out);
////////////////////////////////////////////////////////////////////////////////////////////////////////
	for(int i=0;i<6;i++) delete alpha[i]; 
	delete alpha;

	for(int i=0;i<6;i++) delete beta[i]; 
	delete beta;

	for(int i=0;i<6;i++) delete gama[i]; 
	delete gama; 

	for(int i=0;i<10000;i++) delete buffer_to_int[i]; 
	delete buffer_to_int; 

	for(int i=0;i<49;i++){
		for(int j=0;j<6;j++) delete ems[i][j]; 
	}
	for(int i=0;i<49;i++) delete ems[i]; 
	delete ems;
	
	for(int i=0;i<49;i++){
		for(int j=0;j<6;j++) delete sigma_ems[i][j]; 
	}
	for(int i=0;i<49;i++) delete sigma_ems[i]; 
	delete sigma_ems;

	for(int i=0;i<6;i++) delete sigma_gama[i]; 
	delete sigma_gama; 

	delete sigma_gama_k;
}