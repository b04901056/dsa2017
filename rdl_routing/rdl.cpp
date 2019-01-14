#include <bits/stdc++.h>    
using namespace std;  

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
		source = x;
		target = y;  
		io = i;
		pad = p;
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
	vector< vector<string> > result;
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
		for(int i=0;i<pad_string.size();i++){
			for(int j=0;j<pad_string[i].size();j++){
				cout<<pad_string[i][j]<<" ";
			}
			cout<<endl;
		}
		cout<<endl;
		vector<string> io_connected_pad;
		for(int i=0;i<io_string.size();i++){
			for(int j=0;j<net_list.size();j++){
				if(net_list[j].io == io_string[i]){
					io_connected_pad.push_back(net_list[j].pad);
					break;
				}
			}
		}
		for(int i=0;i<io_connected_pad.size();i++){
			cout<<io_connected_pad[i]<<" ";
		}
		cout<<endl;

		for(int i=1;i<pad_string.size();i++){
			lcs lcs_object(io_connected_pad,pad_string[i]);
			LCS_string.push_back(lcs_object.LCS());
			cout<<endl;
		}
		for(int i=0;i<LCS_string.size();i++){
			for(int j=0;j<LCS_string[i].size();j++){
				cout<<LCS_string[i][j]<<" ";
			}
			cout<<endl;
		}
	}
}; 

int main(int argc, char** argv){
	rdl rdl_object;
	rdl_object.read(argv[1]);
	rdl_object.compute_pad_string();
	rdl_object.compute_lcs();
	
	return 0;
}












