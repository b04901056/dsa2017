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
	double g_cost , h_cost , f_cost;
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
	    /*cout<<endl;
	    for(int i=y_max;i>=0;i--){
		    for(int j=0;j<=x_max;j++){
		    	cout<<is_empty[j][i]<<" ";
		    }
		    cout<<endl;
		}*/
	    return true;
 	}
 	bool routing(int h, vector< pair<int,int> >* result,pair<int,int> whole_die){ // 0:empty 1:obstacle 2:pin 3:wire  

		//cout<<"routing net "<<net_list[h].name<<endl;

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
				//cout<<(*output)<<endl; 
				while(output->parent!=NULL){
					//cout<<*(output->parent)<<endl; 
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
  	bool routing_diea_area(int** die_area,string name,node s,node t,pair<int,int> max_die,pair<int,int> min_die,pair<int,int> whole_die,vector< pair<int,int> >* result){
 		is_empty = die_area;
 		x_max = max_die.first;
 		y_max = max_die.second;
 		x_min = min_die.first;
 		y_min = min_die.second;
 		net n(name,s,t,is_empty);
 		net_list.push_back(n);
 		bool success = routing(0,result,whole_die);
 		net_list.clear();
 		return success;
 	}
}; 

int main(int argc,char** argv){
	router astar;
	vector< pair<int,int> >* result;
	result = new vector< pair<int,int> >;
	astar.read(argv[1]);
	/*astar.routing(0,result);
	astar.routing(1,result);
	astar.routing(2,result);
	for(int i=0;i<(*result).size();i++){
		cout<<(*result)[i].first<<" "<<(*result)[i].second<<endl;
	}*/
	ofstream filePtr;                       
	filePtr.open(argv[2], ios::out);    
	
	int ** die_area = astar.is_empty;
	int x_max = astar.x_max , y_max = astar.y_max;
	string net_name = astar.net_list[0].name;
	node s = astar.net_list[0].source;
	node t = astar.net_list[0].target; 
	router tmp;
	filePtr<<"routing net "<<net_name<<endl;
	bool success = tmp.routing_diea_area(die_area,net_name,s,t,make_pair(x_max,y_max),make_pair(0,0),make_pair(x_max,y_max),result);
	if(success) filePtr<<"Success"<<endl;
	else filePtr<<"Fail"<<endl;
	for(int i=0;i<(*result).size();i++){
		filePtr<<(*result)[i].first<<" "<<(*result)[i].second<<endl;
	}

	(*result).clear(); 
	net_name = astar.net_list[1].name;
	s = astar.net_list[1].target;
	t = astar.net_list[1].source; 
	filePtr<<"routing net "<<net_name<<endl;
	success = tmp.routing_diea_area(die_area,net_name,s,t,make_pair(x_max,y_max),make_pair(0,0),make_pair(x_max,y_max),result);
	if(success) filePtr<<"Success"<<endl;
	else filePtr<<"Fail"<<endl;
	for(int i=0;i<(*result).size();i++){
		filePtr<<(*result)[i].first<<" "<<(*result)[i].second<<endl;
	}

	(*result).clear(); 
	net_name = astar.net_list[2].name;
	s = astar.net_list[2].source;
	t = astar.net_list[2].target;  
	filePtr<<"routing net "<<net_name<<endl;
	success = tmp.routing_diea_area(die_area,net_name,s,t,make_pair(x_max,y_max),make_pair(0,0),make_pair(x_max,y_max),result);
	if(success) filePtr<<"Success"<<endl;
	else filePtr<<"Fail"<<endl;
	for(int i=0;i<(*result).size();i++){
		filePtr<<(*result)[i].first<<" "<<(*result)[i].second<<endl;
	}

	filePtr.close(); 
}