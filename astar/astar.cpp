#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib> 
#include <cstdio> 
#include <cstring>
#include <cmath> 
#include <climits>
using namespace std;
 
class node{
public:
	node(){ 
	}
	node(int x , int y){
		pos[0] = x;
		pos[1] = y;
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
	friend ostream& operator<<(ostream& out, const node& n) {
  		out << "(" << n.pos[0] << " , " << n.pos[1] << " ) "<<endl;;
		return out;
	}
	int pos[2];
};

class wire{
public:
	wire(node x , node y , int** is_empty){
		start = x;
		end = y;
		is_empty[start[0]][start[1]] = 2; 
		is_empty[end[0]][end[1]] = 2;
	}
	node start , end;
	string net_name;
};

class net{
public:
	net(string n , node x , node y , int** is_empty){
		name = n;
		source = x;
		target = y; 
		is_empty[source[0]][source[1]] = 2;
		is_empty[target[0]][target[1]] = 2;
	}
	string name;
	node source , target;
	vector<wire> wire_list;
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
 

class router{
public:
	router(){
	}
	int** is_empty;
 	vector<net> net_list;
 	vector<obstacle> obstacle_list;
 	int x_max , y_max;

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
	        //cout<<line<<endl;
	        pch = strtok(line,delim);
		    while (pch != NULL)
		    {
		      printf ("%s\n",pch);
		      buffer.push_back(pch);
		      pch = strtok (NULL, delim);
		      //printf ("%s\n",pch);

		    } 
		    if(strcmp(buffer[0],"DIEAREA")==0){ 
		    	x_max = atoi(buffer[1]);
		    	y_max = atoi(buffer[2]);
		    	is_empty = new int* [x_max+1];
		    	for(int i=0;i<=x_max;i++){
		    		is_empty[i] = new int[y_max+1];
		    	} 
		    	for(int i=y_max;i>=0;i--){
		    		for(int j=0;j<=x_max;j++){
		    			is_empty[j][i] = 0;
		    		}
		    	} 
		    }
		    else{
		    	int bottom_left_x = atoi(buffer[2]);
		    	int bottom_left_y = atoi(buffer[3]);
		    	int top_right_x = atoi(buffer[4]);
		    	int top_right_y = atoi(buffer[5]); 
		    	if(strcmp(buffer[0],"NET")==0){
		    		net n(buffer[1] , node(bottom_left_x,bottom_left_y) , node(top_right_x,top_right_y) , is_empty);
		    		net_list.push_back(n); 
		    	}
		    	if(strcmp(buffer[0],"OBSTACLE")==0){
		    		obstacle obs(buffer[1] , node(bottom_left_x,bottom_left_y) , node(top_right_x,top_right_y) , is_empty);
		    		obstacle_list.push_back(obs);
		    	}
		    } 
		    buffer.clear();
	    }
	    cout<<endl;
	    for(int i=y_max;i>=0;i--){
		    for(int j=0;j<=x_max;j++){
		    	cout<<is_empty[j][i]<<" ";
		    }
		    cout<<endl;
		}
	    return true;
 	}

 	void routing(){
 		for(int i=0;i<net_list.size();i++){
 			char hold;
 			node current_node , candidate_node , best_node , source_node = net_list[i].source , target_node = net_list[i].target; 
 			vector<node> selected_node;
 			int cost_value = INT_MAX , g_value = 0 , h_value = 0; 
 			current_node = source_node;
 			while(!(current_node == target_node)){
 				cost_value = INT_MAX;
 				for(int j=-1;j<=1;j++){
 					for(int k=-1;k<=1;k++){
 						if((j==0 && k==0) || (j * k) != 0 || (current_node[0] + j) < 0 || (current_node[0] + j) > x_max || (current_node[1] + k) < 0 || (current_node[1] + k) > y_max) continue; 
 						candidate_node = node(current_node[0] + j , current_node[1] + k);
 						cout<<"candidate_node : "<<candidate_node<<endl;
 						cout<<"target_node : "<<target_node<<endl;
 						if(candidate_node == target_node) cost_value = 0;  
 						if(is_empty[current_node[0] + j][current_node[1] + k] != 0 ) continue; 
 						cout<<"candidate_node : "<<candidate_node<<endl;
 						g_value = source_node.manhattan_distance(candidate_node);
 						h_value = candidate_node.manhattan_distance(target_node);
 						if((g_value + h_value) < cost_value){
 							best_node = candidate_node;
 							cost_value = g_value + h_value;
 						}
 					}
 				}
 				cout<<best_node<<endl;
 				selected_node.push_back(current_node); 
 				if(cost_value == 0 ){
 					selected_node.push_back(target_node);
 					net_list[i].node_path = selected_node;
 					break;
 				}
 				else if(cost_value == INT_MAX) break;
				current_node = best_node;
				if(is_empty[best_node[0]][best_node[1]] == 0){
					is_empty[best_node[0]][best_node[1]] = 3;
				} 
 				cout<<"================="<<endl;
 				for(int j=y_max;j>=0;j--){
				    for(int k=0;k<=x_max;k++){
				    	cout<<is_empty[k][j]<<" ";
				    }
				    cout<<endl;
				} 
				cin>>hold;
 			}
 		}
 	}
};

int main(){
	router astar;
	astar.read("test_data.txt"); 
	node a(10,3) , b(1,2);
	//cout<<(a==b)<<endl; 
	//cout<<a.manhattan_distance(b)<<endl;
	//cout<<a<<endl;
	astar.routing();
	cout<<endl;
	for(int i=0;i<astar.net_list.size();i++){
		for(int j=0;j<astar.net_list[i].node_path.size();j++){
			cout<<astar.net_list[i].node_path[j]<<endl;
		} 
		cout<<"============="<<endl;
	}

}
