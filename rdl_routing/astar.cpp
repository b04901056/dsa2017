#include <bits/stdc++.h>
using namespace std;
 
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
	net(string n , node x , node y , int** is_empty){
		name = n;
		source = x;
		target = y; 
		is_empty[source[0]][source[1]] = 2;
		is_empty[target[0]][target[1]] = 2;
	}
	string name;
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
 	bool routing(int h, vector< pair<int,int> >* result){ // 0:empty 1:obstacle 2:pin 3:wire  

		cout<<"routing net "<<net_list[h].name<<endl;

		node source_node = net_list[h].source , target_node = net_list[h].target; 
		priority_queue< node*, vector<node*>, CmpNodePtrs > open;  
		
		// Save the closed_set
		node*** closed_set;
		closed_set = new node** [x_max-x_min+1];
    	for(int i=0;i<=x_max-x_min;i++){
    		closed_set[i] = new node*[y_max-y_min+1];
    	} 
    	for(int i=y_max-y_min;i>=0;i--){
    		for(int j=0;j<=x_max-x_min;j++){
    			closed_set[j][i] = NULL;
    		}
    	} 

		// Record the best distance to a node
		int** shortest_value;
		shortest_value = new int* [x_max-x_min+1];
    	for(int i=0;i<=x_max-x_min;i++){
    		shortest_value[i] = new int[y_max-y_min+1];
    	} 
    	for(int i=y_max-y_min;i>=0;i--){
    		for(int j=0;j<=x_max-x_min;j++){
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
					candidate->g_cost = current->g_cost + 1 ;
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
			/*cout<<"===================="<<endl;
			for(int i=y_max;i>=0;i--){
			    for(int j=0;j<=x_max;j++){
			    	if(is_empty[j][i]!=0){
			    		cout<<setw(10)<<is_empty[j][i]<<" ";
			    	}
			    	else{
			    		cout<<setw(10)<<shortest_value[j][i]<<" ";
			    	}
			    }
			    cout<<endl;
			}
			cout<<"===================="<<endl;*/
		} 
		return false;
 	}
  	bool routing_diea_area(int** die_area,string name,node s,node t,int x_ma,int y_ma,int x_mi,int y_mi,vector< pair<int,int> >* result ){
 		is_empty = die_area;
 		x_max = x_ma;
 		y_max = y_ma;
 		x_min = x_mi;
 		y_min = y_mi;
 		net n(name,s,t,is_empty);
 		net_list.push_back(n);
 		routing(0,result);
 	}
}; 

int main(int argc,char** argv){
	router astar;
	vector< pair<int,int> >* result;
	result = new vector< pair<int,int> >;
	astar.read(argv[1]);
	astar.routing(0,result);
	astar.routing(1,result);
	astar.routing(2,result);
	for(int i=0;i<(*result).size();i++){
		cout<<(*result)[i].first<<" "<<(*result)[i].second<<endl;
	}
	/*
	int ** die_area = astar.is_empty;
	int x_max = astar.x_max , y_max = astar.y_max;
	string net_name = astar.net_list[1].name;
	node s = astar.net_list[1].source;
	node t = astar.net_list[1].target;
	router tmp;
	tmp.routing_diea_area(die_area,net_name,s,t,x_max,y_max,0,0,result);
	for(int i=0;i<(*result).size();i++){
		cout<<(*result)[i].first<<" "<<(*result)[i].second<<endl;
	}*/
}