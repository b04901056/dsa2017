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
	void execute_rdl(int node_num,int* table){
		node_number = node_num;
		pair_table = table;
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
		execute();	
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
		for(int i=0;i<valid;i++) (*record).erase((*record).begin());
		return record;
	}
};

class lcs{
public:
	lcs(vector<string> p,vector<string> q){
		v1 = p;
		v2 = q; 
		n1 = p.size();
		n2 = q.size();
	}
	void print_LCS(int i, int j,vector<string>* result){ 
	    if (i == 0 || j == 0) return;
	 
	    if (prev[i][j] == 0){
	        print_LCS(i-1, j-1, result);
	        (*result).push_back(v1[i-1]);
	    }
	    else if (prev[i][j] == 1) print_LCS(i, j-1, result);
	    else if (prev[i][j] == 2) print_LCS(i-1, j, result);
	}
	vector<string> LCS(){
	    for (int i=0; i<=n1; i++) length[i][0] = 0;
	    for (int j=0; j<=n2; j++) length[0][j] = 0; 
	    for (int i=1; i<=n1; i++){
	        for (int j=1; j<=n2; j++){
	            if (v1[i-1] == v2[j-1]){
	                length[i][j] = length[i-1][j-1] + 1;
	                prev[i][j] = 0;  
	            }
	            else{
	                if (length[i-1][j] < length[i][j-1]){
	                    length[i][j] = length[i][j-1];
	                    prev[i][j] = 1;  
	                }
	                else{
	                    length[i][j] = length[i-1][j];
	                    prev[i][j] = 2;  
	                }
	            }
	        } 
	    }  
	    vector<string>* result;
	    result = new vector<string>;
	    print_LCS(n1, n2, result);
		return *result;
	}
	vector<string> v1,v2;
	int n1, n2; 
	int length[1000][1000];
	int prev[1000][1000]; 

};

class node{
public:
	node(){  
	}
	node(int x , int y){
		pos[0] = x;
		pos[1] = y; 
		parent = NULL; 
	}
	int operator[](int x){
		return pos[x];
	}
	bool operator==(const node& x){
		return ((pos[0]==x.pos[0])&&(pos[1]==x.pos[1]));
	}  
	int manhattan_distance(const node& x){
		return abs(pos[0]-x.pos[0]) + abs(pos[1]-x.pos[1]); 
	}
	void set_f_cost(){
		f_cost = g_cost + h_cost;
	}
	void set_position(int x , int y){
		pos[0] = x;
		pos[1] = y; 
	}
	friend ostream& operator<<(ostream& out, const node& n) {
  		out << "(" << n.pos[0] << " , " << n.pos[1] << " ) ";
		return out;
	}
	int pos[2];
	int g_cost , h_cost , f_cost;
	node* parent;
}; 

class net{
public:
	net(string n,string i,string p,node x,node y){
		name = n;
		source = y;
		target = x;  
		io = i;
		pad = p;
		node_path.push_back(source);
	}
	string name,io,pad;
	node source , target; 
	vector<node> node_path; 
};

class rdl{
public:
	rdl(){

	} 
	string str;  
	fstream file; 
	vector<net> net_list;
	int x_max,y_max;
	int space_between_pad,number_of_pad;
	vector<string> io_string;
	map<string , pair<int,int> > io_position;

	vector<string> bump_pad_list;
	map<string , pair<int,int> > bump_pad_position;
	vector< vector<string> > pad_string;

	vector< vector<string> > LCS_string;
	vector< vector<string> > result_string;

	vector<string> io_connected_pad;

	int** die_area; // 0:empty 1:bump_pad 2:io 3:wire 

	bool read(const string& txt_file){  
		fstream fin;
 		char line[1000];
 		char * delim = " ";
		char * pch; 
		vector<char*> buffer; 
	    fin.open(txt_file,ios::in);
	    if(!fin){
	    	cout<<"input file failure!"<<endl;
	    	return false;
	    }
	    while(fin.getline(line,sizeof(line),'\n')){
	        pch = strtok(line,delim);
		    while (pch != NULL){
		      buffer.push_back(pch);
		      pch = strtok (NULL, delim);
		    }   

		    if(strcmp(buffer[0],"DIEAREA")==0){ 
 				x_max = atoi(buffer[1]);
 				y_max = atoi(buffer[2]);
 				die_area = new int* [x_max+1];
		    	for(int i=0;i<=x_max;i++){
		    		die_area[i] = new int[y_max+1];
		    	} 
		    	for(int i=y_max;i>=0;i--){
		    		for(int j=0;j<=x_max;j++){
		    			die_area[j][i] = 0;
		    		}
		    	} 
		    }
		    else if(strcmp(buffer[0],"IO")==0){
  				io_string.push_back(buffer[1]);
  				io_position[buffer[1]] = make_pair(atoi(buffer[2]),atoi(buffer[3]));
  				die_area[atoi(buffer[2])][atoi(buffer[3])] = 2;
		    } 
		    else if(strcmp(buffer[0],"BUMP")==0){  
   				bump_pad_list.push_back(buffer[1]);
   				bump_pad_position[buffer[1]] = make_pair(atoi(buffer[2]),atoi(buffer[3])); 
   				die_area[atoi(buffer[2])][atoi(buffer[3])] = 2; 
		    } 
		    else if(strcmp(buffer[0],"NET")==0){ 
		    	node a(io_position[buffer[2]].first,io_position[buffer[2]].second);
		    	node b(bump_pad_position[buffer[3]].first,bump_pad_position[buffer[3]].second); 
  				net n(buffer[1],buffer[2],buffer[3],a,b);
		    	net_list.push_back(n);   
		    }  
		    buffer.clear();
	    } 

	    reverse(io_string.begin(),io_string.end());
	    /*for(int i=0;i<net_list.size();i++){
	    	cout<<net_list[i].io<<" "<<net_list[i].pad<<endl;
	    }

	    for(int i=0;i<io_string.size();i++){
	    	cout<<io_string[i]<<" ";
	    }
	    cout<<endl;

	    for(int i=0;i<bump_pad_list.size();i++){
	    	cout<<bump_pad_list[i]<<" ";
	    }
	    cout<<endl; */   

	    for(int i=y_max;i>=0;i--){
		    for(int j=0;j<=x_max;j++){
		    	cout<<die_area[j][i]<<" ";
		    }
		    cout<<endl;
		}     
	    return true;
	}
	void compute_pad_string(){ 
		number_of_pad = pow(bump_pad_list.size(),0.5);
		int tmp = number_of_pad;
		vector<int> ring_pad_number;
		while(tmp > 0){
			ring_pad_number.push_back(tmp);
			tmp-= 2;
		} 
		string start;
		pair<int,int> cur = make_pair(INT_MAX,0);
		for(auto iter = bump_pad_position.begin();iter != bump_pad_position.end();iter++){
			if((iter->second).first <= cur.first && (iter->second).second >= cur.second){
				start = iter->first;
				cur = iter->second;
			}
		}
		int x_pos = cur.first;
		int y_pos = cur.second; 
		space_between_pad = abs(bump_pad_position[bump_pad_list[0]].first - bump_pad_position[bump_pad_list[1]].first ) + abs(bump_pad_position[bump_pad_list[0]].second - bump_pad_position[bump_pad_list[1]].second ); 
		
		for(int i=0;i<ring_pad_number.size();i++){
			vector<string> tmp;
			for(int j=0;j<ring_pad_number[i] - 1;j++){
				x_pos += space_between_pad;
				for(auto iter = bump_pad_position.begin();iter != bump_pad_position.end();iter++){
					if((iter->second).first == x_pos && (iter->second).second == y_pos){
						tmp.push_back(iter->first);
					}
				}
			}
			for(int j=0;j<ring_pad_number[i] - 1;j++){
				y_pos -= space_between_pad;
				for(auto iter = bump_pad_position.begin();iter != bump_pad_position.end();iter++){
					if((iter->second).first == x_pos && (iter->second).second == y_pos){
						tmp.push_back(iter->first);
					}
				}
			}
			for(int j=0;j<ring_pad_number[i] - 1;j++){
				x_pos -= space_between_pad;
				for(auto iter = bump_pad_position.begin();iter != bump_pad_position.end();iter++){
					if((iter->second).first == x_pos && (iter->second).second == y_pos){
						tmp.push_back(iter->first);
					}
				}
			}
			for(int j=0;j<ring_pad_number[i] - 1;j++){
				y_pos += space_between_pad;
				for(auto iter = bump_pad_position.begin();iter != bump_pad_position.end();iter++){
					if((iter->second).first == x_pos && (iter->second).second == y_pos){
						tmp.push_back(iter->first);
					}
				}
			} 
			if(tmp.size() == 0){
				for(auto iter = bump_pad_position.begin();iter != bump_pad_position.end();iter++){
					if((iter->second).first == x_pos && (iter->second).second == y_pos){
						tmp.push_back(iter->first);
					}
				}	
			}
			pad_string.push_back(tmp);
			x_pos += space_between_pad;
			y_pos -= space_between_pad;
		}
		reverse(pad_string.begin(),pad_string.end()); 
	}
	void compute_lcs(){
		cout<<"pad_string:"<<endl;
		for(int i=0;i<pad_string.size();i++){
			for(int j=0;j<pad_string[i].size();j++){
				cout<<pad_string[i][j]<<" ";
			}
			cout<<endl; 
		}
		cout<<endl;  
		for(int i=0;i<io_string.size();i++){
			for(int j=0;j<net_list.size();j++){
				if(net_list[j].io == io_string[i]){
					io_connected_pad.push_back(net_list[j].pad);
					break;
				}
			}
		}
		cout<<"io_string:"<<endl;
		for(int i=0;i<io_connected_pad.size();i++){
			cout<<io_connected_pad[i]<<" ";
		}
		cout<<endl<<endl;

		for(int i=1;i<pad_string.size();i++){
			lcs lcs_object(io_connected_pad,pad_string[i]);
			LCS_string.push_back(lcs_object.LCS()); 
		}

		cout<<"LCS:"<<endl;
		for(int i=0;i<LCS_string.size();i++){
			for(int j=0;j<LCS_string[i].size();j++){
				cout<<LCS_string[i][j]<<" ";
			}
			cout<<endl;
		}
		cout<<endl;
	}
	void compute_net_sequence(){ 
		result_string.push_back(pad_string[0]);

		for(int i=1;i<pad_string.size();i++){
			map<string,int> present_detour_source;
			vector<string> tmp; 
			int pad_current_point = 0 , io_current_point = 0 , tmp_current_point = 0;
  
			// iteration
			for(int j=0;j<LCS_string[i-1].size();j++){ 
				int count = 0;
				while(io_connected_pad[io_current_point] != LCS_string[i-1][j]){ 
					auto  it = find(pad_string[i].begin(), pad_string[i].end(), io_connected_pad[io_current_point]);
					auto  it_ = find(result_string[i-1].begin(), result_string[i-1].end(), io_connected_pad[io_current_point]);
					if(it != pad_string[i].end() || it_ != result_string[i-1].end()){
						tmp.push_back(io_connected_pad[io_current_point]); 
						count++;
					}	 
					io_current_point++;
				}  
				for(int j=0;j<tmp.size();j++){
					cout<<tmp[j]<<" ";
				}
				cout<<endl<<endl;  
				while(pad_string[i][pad_current_point] != LCS_string[i-1][j]){  
					auto  it = find(io_connected_pad.begin(), io_connected_pad.end(), pad_string[i][pad_current_point]);
					if(it != io_connected_pad.end()){
						tmp.push_back(pad_string[i][pad_current_point]);  
						present_detour_source[pad_string[i][pad_current_point]] = tmp.size() - 1;
					}	 
					pad_current_point++; 
				}  
				for(int j=0;j<tmp.size();j++){
					cout<<tmp[j]<<" ";
				}
				cout<<endl<<endl;
				if(tmp_current_point + count != tmp.size()){
					for(int k=0;k<count;k++){
						tmp.push_back(tmp[tmp_current_point + k]);
					}
				} 
				for(int j=0;j<tmp.size();j++){
					cout<<tmp[j]<<" ";
				}
				cout<<endl<<"============="<<endl;

				tmp_current_point = tmp.size();
				io_current_point++;
				pad_current_point++;
			}
			cout<<endl<<"======================================="<<endl;
			// after the last LCS element
			int count = 0;
			while(io_current_point < io_connected_pad.size()){ 
				auto  it = find(pad_string[i].begin(), pad_string[i].end(), io_connected_pad[io_current_point]);
				if(it != pad_string[i].end()){
					tmp.push_back(io_connected_pad[io_current_point]); 
					count++;
				}	 
				io_current_point++;
			}  
			while(pad_current_point < pad_string[i].size()){  
				auto  it = find(io_connected_pad.begin(), io_connected_pad.end(), pad_string[i][pad_current_point]);
				auto  it_ = find(result_string[i-1].begin(), result_string[i-1].end(), io_connected_pad[io_current_point]);
				if(it != pad_string[i].end() || it_ != result_string[i-1].end()){
					tmp.push_back(pad_string[i][pad_current_point]);  
					present_detour_source[pad_string[i][pad_current_point]] = tmp.size() - 1;
				}	 
				pad_current_point++; 
			}  
			if(tmp_current_point + count != tmp.size()){
				for(int j=0;j<count;j++){
					tmp.push_back(tmp[tmp_current_point + j]);
				}
			} 

			for(int j=0;j<tmp.size();j++){
				cout<<tmp[j]<<" ";
			}
			cout<<endl;

			// construct mpsc object
			vector< pair<int,int> > mpsc_pair;
			map<string,int> present_detour_num; 

			for(auto it = present_detour_source.begin();it != present_detour_source.end();it++){
				cout<<it->first<<" : "<<it->second<<"  ";
				present_detour_num[it->first] = -2;
			}
			cout<<endl;
			for(int j=0;j<tmp.size();j++){
				if(present_detour_source.find(tmp[j]) != present_detour_source.end()) present_detour_num[tmp[j]] += 1;
			}
			for(auto it = present_detour_num.begin();it != present_detour_num.end();it++){
				cout<<it->first<<" : "<<it->second<<"  ";
			}
			cout<<endl;
			for(auto it = present_detour_num.begin();it != present_detour_num.end();it++){
				for(int k=0;k<it->second;k++){
					tmp.insert(tmp.begin() + present_detour_source[it->first],it->first);
					for(auto iterator = present_detour_source.begin();iterator != present_detour_source.end();iterator++){
						if(iterator->second >= present_detour_source[it->first]) iterator->second++;
					}
				}
			}
			for(auto it = present_detour_num.begin();it != present_detour_num.end();it++){
				int start = 0,end = tmp.size()-1;
				while(true){ 
					while(tmp[start] != it->first) start++;
					while(tmp[end] != it->first) end--;
					if(start >= end) break;
					mpsc_pair.push_back( make_pair(start,end) ); 
					start++;
					end--;
				}
			}
			for(int j=0;j<mpsc_pair.size();j++){
				cout<<mpsc_pair[j].first<<" "<<mpsc_pair[j].second<<endl;
			}
			cout<<endl;
			for(int j=result_string[i-1].size()-1;j>=0;j--){
				int term = tmp.size();
				for(int k=0;k<term;k++){
					if(result_string[i-1][j] == tmp[k]){
						mpsc_pair.push_back( make_pair(k,tmp.size()) );
						tmp.push_back(result_string[i-1][j]);
						present_detour_source[tmp[k]] = tmp.size() - 1;
					}
				}
			}

			for(int j=0;j<mpsc_pair.size();j++){
				cout<<mpsc_pair[j].first<<" "<<mpsc_pair[j].second<<endl;
			} 
			cout<<endl;

			// calculate result_string 
			int* pair_table;
			pair_table = new int[mpsc_pair.size()*2];
			for(int j=0;j<mpsc_pair.size();j++){
				pair_table[mpsc_pair[j].first] = mpsc_pair[j].second;
				pair_table[mpsc_pair[j].second] = mpsc_pair[j].first;
			}
			mpsc mpsc_object;
			mpsc_object.execute_rdl(mpsc_pair.size()*2,pair_table);
			vector<node_pair>* mis = mpsc_object.result();

			for(int j=0;j<(*mis).size();j++){
				cout<<(*mis)[j].first<<" "<<(*mis)[j].second<<endl;
			}
			cout<<endl;

			// 選 present_detour_source 的另外一個 !

			// update net path


		}
	}
}; 

int main(int argc, char** argv){ 
	rdl rdl_object;
	rdl_object.read(argv[1]);
	rdl_object.compute_pad_string();
	rdl_object.compute_lcs();
	rdl_object.compute_net_sequence();
	
	return 0;
}












