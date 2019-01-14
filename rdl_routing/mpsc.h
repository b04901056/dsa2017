#include <bits/stdc++.h>     
using namespace std;

typedef pair<int,int> node_pair;

int Partition(vector<node_pair>* A, int p, int r);
void QuickSort(vector<node_pair>* A, int p, int r);

class mis{
public:
	mis();
	mis* back_tracing_first;
	mis* back_tracing_second;
	node_pair present;
	int cardinality;
};
 

void back_track(mis* x,vector<node_pair>* record);

class mpsc{
public:
	mpsc();
	mis*** mpsc_table;
	int* pair_table;
	int node_number = 0;
	int node_i , node_j;  
	string str;  
	fstream file; 

	bool read(const string& txt_file);

	void execute();

	vector<node_pair>* result();
}; 