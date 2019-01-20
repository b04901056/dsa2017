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
	node(pair<int,int> p){
		pos[0] = p.first;
		pos[1] = p.second; 
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
	double g_cost , h_cost , f_cost;
	node* parent;
}; 


class net{
public:
	net(string n , string i , string p , node x , node y , int** is_empty){
		name = n;
		source = x;
		target = y; 
		io = i;
		pad = p; 
		is_empty[source[0]][source[1]] = 2;
		is_empty[target[0]][target[1]] = 2;
	}
	net(string n , node x , node y , int** is_empty){
		name = n;
		source = x;
		target = y;  
		is_empty[source[0]][source[1]] = 2;
		is_empty[target[0]][target[1]] = 2;
	} 
	string name,io,pad;
	node source , target; 
	vector<node> node_path; 
}; 

class obstacle{
public:
	obstacle(string n , node a , node b , int** is_empty){
		name = n;
		bottom_left = a;
		top_right = b;
		for(int i=bottom_left[0];i<=top_right[0];i++){
			for(int j=bottom_left[1];j<=top_right[1];j++){
				is_empty[i][j] = 1;
			}
		}
	}
	string name;
	node bottom_left , top_right; 
};
 
struct CmpNodePtrs
{
    bool operator()(const node* lhs, const node* rhs) const
    {
        return lhs->f_cost > rhs->f_cost;
    }
};

class router{
public:
	router(){
		x_min = 0;
		y_min = 0;
	}
	int** is_empty;
 	vector<net> net_list;
 	vector<obstacle> obstacle_list;
 	int x_max,y_max,x_min,y_min;

 	bool routing(int h, vector< pair<int,int> >* result,pair<int,int> whole_die){ // 0:empty 1:obstacle 2:pin 3:wire  

		cout<<"routing net "<<net_list[h].name<<endl;

		node source_node = net_list[h].source , target_node = net_list[h].target; 
		priority_queue< node*, vector<node*>, CmpNodePtrs > open;  
		
		// Save the closed_set
		node*** closed_set;
		closed_set = new node** [whole_die.first+1];
    	for(int i=0;i<=whole_die.first;i++){
    		closed_set[i] = new node*[whole_die.second+1];
    	} 
    	for(int i=whole_die.second;i>=0;i--){
    		for(int j=0;j<=whole_die.first;j++){
    			closed_set[j][i] = NULL;
    		}
    	} 

		// Record the best distance to a node
		double** shortest_value;
		shortest_value = new double* [whole_die.first+1];
    	for(int i=0;i<=whole_die.first;i++){
    		shortest_value[i] = new double[whole_die.second+1];
    	} 
    	for(int i=whole_die.second-y_min;i>=0;i--){
    		for(int j=0;j<=whole_die.first;j++){
    			shortest_value[j][i] = INT_MAX;
    		}
    	}    
		// Initialization
		node* init_node = new node;
		*init_node = source_node; 
		init_node->g_cost = 0;
		init_node->h_cost = init_node->manhattan_distance(target_node); 
		init_node->set_f_cost();

		is_empty[source_node[0]][source_node[1]] = 0;  
		is_empty[target_node[0]][target_node[1]] = 0;  

		//closed_set[init_node->pos[0]][init_node->pos[1]] = init_node;
		shortest_value[init_node->pos[0]][init_node->pos[1]] = init_node->f_cost;
		open.push(init_node); 

		// Searching
		while(!open.empty()){ 
			node* current = new node; 
			while(true){
				if(open.empty()) break;
				current = open.top(); 
				open.pop();	 
				//if(closed_set[current->pos[0]][current->pos[1]] != NULL) continue;				//?
				if(current->f_cost <= shortest_value[current->pos[0]][current->pos[1]]) break;
			}  
			//cout<<"Move node "<<(*current)<<" to the closed_set"<<endl; 
			closed_set[current->pos[0]][current->pos[1]] = current;  

			// Termination
			if(*current == target_node){   
				node* output = current;  
				(*result).push_back(make_pair(output->pos[0],output->pos[1])); 
				cout<<(*output)<<endl; 
				while(output->parent!=NULL){ 
					cout<<*(output->parent)<<endl; 
					output = output->parent;
					(*result).push_back(make_pair(output->pos[0],output->pos[1]));
					is_empty[output->pos[0]][output->pos[1]] = 3; 
				}
				is_empty[source_node[0]][source_node[1]] = 2; 
				is_empty[target_node[0]][target_node[1]] = 2; 
				return true;
			}

			// Go through neighbors
			for(int j=-1;j<=1;j++){
				for(int k=-1;k<=1;k++){  
					// If candidate is not traversable or in closed_set => skip  
					if((j==0 && k==0)) continue;   
					if((current->pos[0]+j < x_min ) || (current->pos[0]+j > x_max ) || (current->pos[1]+k < y_min)|| (current->pos[1]+k > y_max)) continue;
					if( (j * k != 0) && (is_empty[current->pos[0] + j ][current->pos[1]] != 0 && is_empty[current->pos[0]][current->pos[1] + k ] != 0)) continue; 

					node* candidate = new node(current->pos[0]+j , current->pos[1]+k);  
					if(k*j == 0) candidate->g_cost = current->g_cost + 1 ;
					else candidate->g_cost = current->g_cost + pow(2,0.5) ;
					candidate->h_cost = candidate->manhattan_distance(target_node);
					candidate->set_f_cost(); 

					if(closed_set[candidate->pos[0]][candidate->pos[1]] != NULL) continue;
					if(is_empty[candidate->pos[0]][candidate->pos[1]] != 0 ) continue;

					// If new path to the candidate is shorter or candidate not in open set => update
					candidate->parent = current; 
					open.push(candidate);
					//cout<<"Move node "<<(*candidate)<<" to the open_set "<<"f_cost = "<<candidate->f_cost<<endl;
					if(candidate->f_cost < shortest_value[candidate->pos[0]][candidate->pos[1]]){
						shortest_value[candidate->pos[0]][candidate->pos[1]] = candidate->f_cost;
					}
				}
			}   
		} 
		return false;
 	}
  	bool routing_diea_area(int** die_area,string name,node s,node t,pair<int,int> max_die,pair<int,int> min_die,pair<int,int> whole_die,vector< pair<int,int> >* result_path){
 		is_empty = die_area;
 		x_max = max_die.first;
 		y_max = max_die.second;
 		x_min = min_die.first;
 		y_min = min_die.second; 
 		net n(name,s,t,is_empty);
 		net_list.push_back(n);
 		bool success = routing(0,result_path,whole_die);
 		net_list.clear();
 		return success;
 	}
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
	vector<string> io_connected_pad;

	vector< vector<string> > LCS_string;
	vector< vector<string> > result_string;
	vector< vector<string> > mpsc_priority;


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
  				//die_area[atoi(buffer[2])][atoi(buffer[3])] = 2;
		    } 
		    else if(strcmp(buffer[0],"BUMP")==0){  
   				bump_pad_list.push_back(buffer[1]);
   				bump_pad_position[buffer[1]] = make_pair(atoi(buffer[2]),atoi(buffer[3])); 
   				//die_area[atoi(buffer[2])][atoi(buffer[3])] = 2; 
		    } 
		    else if(strcmp(buffer[0],"NET")==0){ 
		    	node a(io_position[buffer[2]].first,io_position[buffer[2]].second);
		    	node b(bump_pad_position[buffer[3]].first,bump_pad_position[buffer[3]].second); 
  				net n(buffer[1],buffer[2],buffer[3],a,b,die_area);
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
		/*cout<<"pad_string:"<<endl;
		for(int i=0;i<pad_string.size();i++){
			for(int j=0;j<pad_string[i].size();j++){
				cout<<pad_string[i][j]<<" ";
			}
			cout<<endl; 
		}*/
		//cout<<endl;  
		for(int i=0;i<io_string.size();i++){
			for(int j=0;j<net_list.size();j++){
				if(net_list[j].io == io_string[i]){
					io_connected_pad.push_back(net_list[j].pad);
					break;
				}
			}
		}
		/*cout<<"io_string:"<<endl;
		for(int i=0;i<io_connected_pad.size();i++){
			cout<<io_connected_pad[i]<<" ";
		}
		cout<<endl<<endl;*/

		for(int i=1;i<pad_string.size();i++){
			lcs lcs_object(io_connected_pad,pad_string[i]);
			LCS_string.push_back(lcs_object.LCS()); 
		}

		/*cout<<"LCS:"<<endl;
		for(int i=0;i<LCS_string.size();i++){
			for(int j=0;j<LCS_string[i].size();j++){
				cout<<LCS_string[i][j]<<" ";
			}
			cout<<endl;
		}
		cout<<endl;*/
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
				/*for(int j=0;j<tmp.size();j++){
					cout<<tmp[j]<<" ";
				}
				cout<<endl<<endl;*/  
				while(pad_string[i][pad_current_point] != LCS_string[i-1][j]){  
					auto  it = find(io_connected_pad.begin(), io_connected_pad.end(), pad_string[i][pad_current_point]);
					if(it != io_connected_pad.end()){
						tmp.push_back(pad_string[i][pad_current_point]);  
						present_detour_source[pad_string[i][pad_current_point]] = tmp.size() - 1;
					}	 
					pad_current_point++; 
				}  
				/*for(int j=0;j<tmp.size();j++){
					cout<<tmp[j]<<" ";
				}
				cout<<endl<<endl;*/
				if(tmp_current_point + count != tmp.size()){
					for(int k=0;k<count;k++){
						tmp.push_back(tmp[tmp_current_point + k]);
					}
				} 
				/*for(int j=0;j<tmp.size();j++){
					cout<<tmp[j]<<" ";
				}
				cout<<endl<<"============="<<endl;*/

				tmp_current_point = tmp.size();
				io_current_point++;
				pad_current_point++;
			}
			//cout<<endl<<"======================================="<<endl;
			// after the last LCS element
			int count = 0;
			while(io_current_point < io_connected_pad.size()){ 
				auto  it = find(pad_string[i].begin(), pad_string[i].end(), io_connected_pad[io_current_point]);
				auto  it_ = find(result_string[i-1].begin(), result_string[i-1].end(), io_connected_pad[io_current_point]);
				if(it != pad_string[i].end() || it_ != result_string[i-1].end()){
					tmp.push_back(io_connected_pad[io_current_point]); 
					count++;
				}	 
				io_current_point++;
			}  
			while(pad_current_point < pad_string[i].size()){  
				auto  it = find(io_connected_pad.begin(), io_connected_pad.end(), pad_string[i][pad_current_point]);
				if(it != io_connected_pad.end()){
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

			/*for(int j=0;j<tmp.size();j++){
				cout<<tmp[j]<<" ";
			}
			cout<<endl;*/

			// construct mpsc object
			vector< pair<int,int> > mpsc_pair;
			map<string,int> present_detour_num; 

			for(auto it = present_detour_source.begin();it != present_detour_source.end();it++){
				//cout<<it->first<<" : "<<it->second<<"  ";
				present_detour_num[it->first] = -2;
			}
			//cout<<endl;
			for(int j=0;j<tmp.size();j++){
				if(present_detour_source.find(tmp[j]) != present_detour_source.end()) present_detour_num[tmp[j]] += 1;
			}
			/*for(auto it = present_detour_num.begin();it != present_detour_num.end();it++){
				cout<<it->first<<" : "<<it->second<<"  ";
			}
			cout<<endl;*/
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
			/*for(int j=0;j<tmp.size();j++){
				cout<<tmp[j]<<" ";
			}
			cout<<endl;

			for(auto it = present_detour_source.begin();it != present_detour_source.end();it++){
				cout<<it->first<<" : "<<it->second<<"  "; 
			}
			cout<<endl;
			for(int j=0;j<mpsc_pair.size();j++){
				cout<<mpsc_pair[j].first<<" "<<mpsc_pair[j].second<<endl;
			} 
			cout<<endl;*/

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

			/*for(int j=0;j<(*mis).size();j++){
				cout<<(*mis)[j].first<<" "<<(*mis)[j].second<<endl;
			}
			cout<<endl;*/

			vector<string> tmp_mpsc;
			for(int j=0;j<(*mis).size();j++){
				tmp_mpsc.push_back(tmp[(*mis)[j].first]);
			}
			mpsc_priority.push_back(tmp_mpsc);

			// 選 present_detour_source 的另外一個 !

			/*for(auto it = present_detour_source.begin();it != present_detour_source.end();it++){
				cout<<it->first<<" : "<<it->second<<"  "; 
			}
			cout<<endl<<"============"<<endl;*/ 

			vector<string> v;
			set<string> not_selected_add;
			for(int j=0;j<tmp.size();j++){
				auto  it = find(result_string[i-1].begin(), result_string[i-1].end(), tmp[j]);
				if(it != result_string[i-1].end()){
					if(present_detour_source[tmp[j]] != j && present_detour_source[tmp[j]] != j+1){ 
						bool add = false, not_selected = true;
						for(int k=0;k<(*mis).size();k++){
							if((*mis)[k].first == j || (*mis)[k].second == j){
								add = true;
								break;
							}
							if(tmp[(*mis)[k].first] == tmp[j] || tmp[(*mis)[k].second] == tmp[j]){
								not_selected = false;
							}
						}
						if(not_selected && not_selected_add.count(tmp[j]) != 0) not_selected = false;
						if(add || not_selected){
							 v.push_back(tmp[j]); 
							 if(not_selected) not_selected_add.insert(tmp[j]);
						}
					}
				}
				else{
					if(present_detour_source[tmp[j]] != j){
						bool add = false, not_selected = true;
						for(int k=0;k<(*mis).size();k++){
							if((*mis)[k].first == j || (*mis)[k].second == j){
								add = true;
								break;
							}
							if(tmp[(*mis)[k].first] == tmp[j] || tmp[(*mis)[k].second] == tmp[j]){
								not_selected = false;
							}
						}
						if(not_selected && not_selected_add.count(tmp[j]) != 0) not_selected = false;
						if(add || not_selected){
							 v.push_back(tmp[j]); 
							 if(not_selected) not_selected_add.insert(tmp[j]);
						}
					}
				} 
			} 
			tmp = v;
			/*for(int j=0;j<tmp.size();j++){
				cout<<tmp[j]<<" ";
			}
			cout<<endl; */

			int lcs_point = 0,tmp_point = 0,io_connected_pad_point = 0;
			while(true){
				if(io_connected_pad[io_connected_pad_point] == LCS_string[i-1][lcs_point]){
					tmp.insert(tmp.begin()+tmp_point,io_connected_pad[io_connected_pad_point]);
					lcs_point++;
					tmp_point++;
				}
				else if(io_connected_pad[io_connected_pad_point] == tmp[tmp_point]){
					tmp_point++;
				}
				io_connected_pad_point++; 
				if(lcs_point == LCS_string[i-1].size() || io_connected_pad_point == io_connected_pad.size()) break; 
			} 

			result_string.push_back(tmp); 
		}
	}
}; 

int main(int argc, char** argv){ 
	rdl rdl_object;
	rdl_object.read(argv[1]);
	rdl_object.compute_pad_string();
	rdl_object.compute_lcs();
	rdl_object.compute_net_sequence();
	cout<<endl<<"==========================================================="<<endl; 
	cout<<"result_string:"<<endl;
	for(int i=0;i<rdl_object.result_string.size();i++){
		for(int j=0;j<rdl_object.result_string[i].size();j++){
			cout<<rdl_object.result_string[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	cout<<"LCS_string"<<endl;
	for(int i=0;i<rdl_object.LCS_string.size();i++){
		for(int j=0;j<rdl_object.LCS_string[i].size();j++){
			cout<<rdl_object.LCS_string[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	cout<<"mpsc_priority:"<<endl;
	for(int i=0;i<rdl_object.mpsc_priority.size();i++){
		for(int j=0;j<rdl_object.mpsc_priority[i].size();j++){
			cout<<rdl_object.mpsc_priority[i][j]<<" ";
		}
		cout<<endl;
	}

	/*router router_rdl;
	vector< pair<int,int> >* result_path;
	result_path = new vector< pair<int,int> >;
	string net_name = "b_14";
	node t(12,10);
	pair<int,int> max,min,whole;
	max = rdl_object.bump_pad_position["b_19"];
	min = rdl_object.bump_pad_position["b_07"]; 
	whole = make_pair(30,30);
	cout<<router_rdl.routing_diea_area(rdl_object.die_area,net_name,node(rdl_object.bump_pad_position[net_name]),t,max,min,whole,result_path)<<endl;
	
	return 0;*/
}












