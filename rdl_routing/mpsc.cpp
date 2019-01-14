#include <bits/stdc++.h>          
using namespace std;

typedef pair<int,int> node_pair;
 
int Partition(vector<node_pair>* A, int p, int r){ 
	node_pair x = (*A)[p]; 
    int i = p - 1 ;
    int j = r + 1 ;
    while(true){
    	j--;
    	while((*A)[j].first > x.first){
    		j--; 
    	}
    	i++; 
    	while((*A)[i].first < x.first){
    		i++;  
    	}  
		if(i < j){ 
			node_pair tmp = (*A)[i];
        	(*A)[i] = (*A)[j];
        	(*A)[j] = tmp; 
		} 
		else return j;
    }
}

void QuickSort(vector<node_pair>* A, int p, int r){ 
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
		cardinality = 0;
	}
	mis* back_tracing_first;
	mis* back_tracing_second;
	node_pair present;
	int cardinality;
};
 

void back_track(mis* x,vector<node_pair>* record){ 
	(*record).push_back(x->present); 
	if(x->back_tracing_first) back_track(x->back_tracing_first,record);
	if(x->back_tracing_second) back_track(x->back_tracing_second,record); 
}

class mpsc{
public:
	mpsc(){

	}
	mis*** mpsc_table;
	int* pair_table;
	int node_number = 0;
	int node_i , node_j;  
	string str;  
	fstream file; 

	bool read(const string& txt_file){
		file.open(txt_file, ios::in); 
		file>>str; 
		node_number = stoi(str);
		pair_table = new int[node_number]; 	
		for(int i=0;i<node_number/2;i++){ 
			file>>str;
			node_i = stoi(str);
			file>>str;
			node_j = stoi(str);
			pair_table[node_i] = node_j;
			pair_table[node_j] = node_i;
		} 
		mpsc_table = new mis**[node_number];
		for(int i=0;i<node_number;i++){
			mpsc_table[i] = new mis*[node_number];
		}

		for(int i=0;i<node_number;i++){
			for(int j=0;j<node_number;j++){
				if(i <= j) mpsc_table[i][j] = new mis();
				else mpsc_table[i][j] = NULL;
			}
		}
	}

	void execute(){
		for(int j = 0 ;j < node_number;j++){
			for(int i = 0 ; i <= (j - 1);i++){ 
				int k = pair_table[j]; 

				if(k==i){ 								//case 3
					if( k + 1 <= j - 1 ){
						if(mpsc_table[k+1][j-1]->cardinality + 1 > mpsc_table[i][j-1]->cardinality){
							mpsc_table[i][j]->present = make_pair(i,j);  
							mpsc_table[i][j]->back_tracing_first = mpsc_table[i+1][j-1]; 
							mpsc_table[i][j]->cardinality = mpsc_table[i+1][j-1]->cardinality + 1;
						}
						else{
							mpsc_table[i][j]->back_tracing_first = mpsc_table[i][j-1];
							mpsc_table[i][j]->cardinality = mpsc_table[i][j-1]->cardinality;
						} 
					}
					else if( k + 1 == j ){ 
						mpsc_table[i][j]->present = make_pair(i,j);   
						mpsc_table[i][j]->cardinality = 1; 
					}
					else{
						mpsc_table[i][j]->back_tracing_first = mpsc_table[i][j-1];
						mpsc_table[i][j]->cardinality = mpsc_table[i][j-1]->cardinality;
					} 
				} 
				else if( (k>i) && (k<j)){				//case 2
					if( k + 1 <= j - 1){
						if(mpsc_table[i][k-1]->cardinality + 1 + mpsc_table[k+1][j-1]->cardinality > mpsc_table[i][j-1]->cardinality){
							mpsc_table[i][j]->present = make_pair(k,j); 
							mpsc_table[i][j]->back_tracing_first = mpsc_table[i][k-1];
							mpsc_table[i][j]->back_tracing_second = mpsc_table[k+1][j-1];
							mpsc_table[i][j]->cardinality = mpsc_table[i][k-1]->cardinality + 1 + mpsc_table[k+1][j-1]->cardinality;
						}
						else{
							mpsc_table[i][j]->back_tracing_first = mpsc_table[i][j-1];
							mpsc_table[i][j]->cardinality = mpsc_table[i][j-1]->cardinality;
						}
					}
					else if(mpsc_table[i][k-1]->cardinality + 1 > mpsc_table[i][j-1]->cardinality){
						mpsc_table[i][j]->present = make_pair(k,j); 
						mpsc_table[i][j]->back_tracing_first = mpsc_table[i][k-1]; 
						mpsc_table[i][j]->cardinality = mpsc_table[i][k-1]->cardinality + 1;
					}
					else{
						mpsc_table[i][j]->back_tracing_first = mpsc_table[i][j-1];
						mpsc_table[i][j]->cardinality = mpsc_table[i][j-1]->cardinality;
					}
				}
				else{									//case 1
					mpsc_table[i][j]->back_tracing_first = mpsc_table[i][j-1];
					mpsc_table[i][j]->cardinality = mpsc_table[i][j-1]->cardinality;
				} 
			}
		}	
	}

	vector<node_pair>* result(){
		mis* x = mpsc_table[0][node_number-1]; 
		vector<node_pair>* record = new vector<node_pair>; 
	 
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
		return record;
	}
};

int main(int argc, char** argv){
	mpsc mpsc_object;
	mpsc_object.read(argv[1]);
	mpsc_object.execute();
	mpsc_object.result(); 
}