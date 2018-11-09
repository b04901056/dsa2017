#include <iostream>
#include <fstream>  
#include <string> 
#include <vector>           
using namespace std;

typedef pair<int,int> node_pair;
 
int Partition(vector<node_pair>* A, int p, int r){
	//cout<<"Partition "<<p<<" "<<r<<endl;
	node_pair x = (*A)[p]; 
    int i = p - 1 ;
    int j = r + 1 ;
    while(true){
    	j--;
    	while((*A)[j].first > x.first){
    		j--;
    		//cout<<"j= "<<j<<endl;
    	}
    	i++; 
    	while((*A)[i].first < x.first){
    		i++; 
    		//cout<<"i= "<<i<<endl;
    	}  
		if(i < j){
			//cout<<"swap "<<i<<" "<<j<<endl;
			//swap(A[i] , A[j]);
			node_pair tmp = (*A)[i];
        	(*A)[i] = (*A)[j];
        	(*A)[j] = tmp; 
		} 
		else return j;
    }
}

void QuickSort(vector<node_pair>* A, int p, int r){
	//cout<<"QuickSort "<<p<<" "<<r<<endl;
    if (p < r) {
        int q = Partition(A, p, r);
        QuickSort(A, p, q);
        QuickSort(A, q + 1, r);
    }
}

class mis{
public:
	mis(){
		back_tracing_first = NULL;
		back_tracing_second = NULL;
		present = make_pair(-1,-1);
	}
	mis* back_tracing_first;
	mis* back_tracing_second;
	node_pair present;
};

mis*** mpsc_table; 

void back_track(mis* x,vector<node_pair>* record){ 
	(*record).push_back(x->present); 
	if(x->back_tracing_first) back_track(x->back_tracing_first,record);
	if(x->back_tracing_second) back_track(x->back_tracing_second,record); 
}

int main(int argc, char** argv){
	int* pair_table;
	int node_number = 0;
	int node_i , node_j;  
	string str;  
	fstream file;

	file.open(argv[1], ios::in); 
	file>>str; 
	node_number = stoi(str);
	pair_table = new int[node_number]; 
	
	// Read in 
	for(int i=0;i<node_number/2;i++){ 
		file>>str;
		node_i = stoi(str);
		file>>str;
		node_j = stoi(str);
		pair_table[node_i] = node_j;
		pair_table[node_j] = node_i;
	}

	for(int i=0;i<node_number;i++){
		cout<<i<<" : "<<pair_table[i]<<endl; 
	}

	// Construct MPSC table
	mpsc_table = new mis**[node_number];
	for(int i=0;i<node_number;i++){
		mpsc_table[i] = new mis*[node_number];
	}

	for(int i=0;i<node_number;i++){
		for(int j=0;j<node_number;j++){
			if(i<=j) mpsc_table[i][j] = new mis();
			else mpsc_table[i][j] = NULL;
		}
	}

	//cout<<"("<<mpsc_table[0][0]->present.first<<","<<mpsc_table[0][0]->present.second<<")"<<endl;  
	
	// Dynamic Programming
	for(int l=1;l<=node_number;l++){
		for(int i=0;i<=(node_number-l);i++){
			int j = i + l - 1;
			if(i==j) continue;
			else{
				int k = pair_table[j];
				if(k==i){ 								//case 3
					mpsc_table[i][j]->present = make_pair(i,j); 
					if( (i+1<node_number) && (j-1>=0)){
						if(mpsc_table[i+1][j-1]) mpsc_table[i][j]->back_tracing_first = mpsc_table[i+1][j-1];
					}
				} 
				else if( (k>i) && (k<j)){	//case 2
					mpsc_table[i][j]->present = make_pair(k,j); 
					if( (i<node_number) && (k-1>=0)){
						if(mpsc_table[i][k-1]) mpsc_table[i][j]->back_tracing_first = mpsc_table[i][k-1];
					}
					if( (k+1<node_number) && (j-1>=0)){
						if(mpsc_table[k+1][j-1]) mpsc_table[i][j]->back_tracing_second = mpsc_table[k+1][j-1];
					}
				}
				else{												//case 1
					if( (i<node_number) && (j-1>=0)){
						if(mpsc_table[i][j-1]) mpsc_table[i][j]->back_tracing_first = mpsc_table[i][j-1];
					}
				}
			}  
		}
	}
	 
	// Show result
	cout<<"------------------"<<endl;
	mis* x = mpsc_table[0][node_number-1]; 

	vector<node_pair>* record = new vector<node_pair>; 

	for(int i=0;i<(*record).size();i++){
		cout<<"("<<(*record)[i].first<<","<<(*record)[i].second<<")"<<endl;
	}

	back_track(x,record);
	QuickSort(record, 0, record->size()-1);

	int valid = 0;
	for(int i=0;i<record->size();i++){
		if((*record)[i].first != -1){
			valid = i;
			break;
		} 
	}
	cout<< record->size() - valid <<endl;
	for(int i=valid;i<record->size();i++){ 
		cout<<(*record)[i].first<<" "<<(*record)[i].second<<endl; 
	}

	// Write out

	ofstream filePtr;                       
	filePtr.open(argv[2], ios::out);      

	filePtr<< record->size() - valid <<endl;
	for(int i=valid;i<record->size();i++){ 
		filePtr<<(*record)[i].first<<" "<<(*record)[i].second<<endl; 
	}

	filePtr.close(); 

	// Delete
	delete record;
	for(int i=0;i<node_number;i++){
		for(int j=0;j<node_number;j++){
			delete mpsc_table[i][j];
		}
	}
	for(int i=0;i<node_number;i++){
		delete[] mpsc_table[i]; 
	}
	delete [] mpsc_table;

	return 0;
}