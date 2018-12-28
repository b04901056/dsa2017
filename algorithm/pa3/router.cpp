#include <bits/stdc++.h>
using namespace std;   

double** weight; 
double base = 10;

class router{
public:
	router(){}
	int x_max,y_max,net_num,MAX_NODE;
	double capacity; 
	pair<int,int>* net_list;
	vector< pair<int,int> >* answer; 

 	bool read(const string& txt_file){
 		fstream fin;
 		char line[1000];
 		char* delim = " ";
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
		    if(strcmp(buffer[0],"grid") == 0){ 
		    	x_max = atoi(buffer[1]);
		    	y_max = atoi(buffer[2]); 
		    	MAX_NODE = x_max * y_max;
		    }
		    else if(strcmp(buffer[0],"capacity") == 0){
		    	capacity = atoi(buffer[1]);
		    } 
		    else if(strcmp(buffer[0],"num") == 0 && strcmp(buffer[1],"net") == 0){
		    	net_num = atoi(buffer[2]);
		    	net_list = new pair<int,int>[net_num];
		    	answer = new vector< pair<int,int> >[net_num];
		    }  
		    else{ 
		    	net_list[atoi(buffer[0])] = make_pair(cooridate_to_id(atoi(buffer[1]),atoi(buffer[2])),cooridate_to_id(atoi(buffer[3]),atoi(buffer[4]))); 
		    }
		    buffer.clear();
	    }

	    // Initialize weight and demand 

	    weight = new double*[MAX_NODE]; 
		for(int i=0;i<MAX_NODE;i++){ 
			weight[i] = new double[4]; 
		} 
		for(int i=0;i<MAX_NODE;i++){ 
			for(int j=0;j<4;j++){
				weight[i][j] = 0.0; 
			}
		}  
 	}
 	 
 	pair<int,int> id_to_cooridate(int p){
 		return make_pair( int(p % x_max) , int(p / x_max));
	}

	int cooridate_to_id(int x,int y){
		return y * x_max + x ; 
	} 

	double weight_to_demand(double w){ 
		return log(w + 1) / log(base) * capacity;
	}
	double demand_to_weight(double d){ 
		return pow(base , d / capacity) - 1;
	}
 
 	void route(int n){ 
 		const int dx[] = {0, 1, 0, -1};
		const int dy[] = {1, 0, -1, 0}; // up right down left

 		int predecessor[MAX_NODE];
		double distance[MAX_NODE]; 

		for(int i=0;i<MAX_NODE;i++){
			predecessor[i] = -1;
			distance[i] = INT_MAX;  
		} 
		distance[net_list[n].first] = 0;

		priority_queue< pair<int, int> > heap; 
		heap.push(make_pair(0, net_list[n].first)); // distance : id

		while (heap.size()) { 
			while (heap.size() && -heap.top().first != distance[heap.top().second])
				heap.pop();	 
			if(heap.size() == 0) break;

			pair<int, int> out = heap.top();
			heap.pop();

			int node = out.second; //id
			int dis = - out.first;

			int x = id_to_cooridate(node).first;
			int y = id_to_cooridate(node).second;
			
			for (int i = 0; i < 4; i++) {
				int X = x + dx[i];
				int Y = y + dy[i];
				int id = cooridate_to_id(X, Y);
				if (X < 0 || X >= x_max || Y < 0 || Y >= y_max) continue;
				
				int new_dis = dis + weight[node][i];
				if (new_dis >= distance[id]) continue;

				distance[id] = new_dis;
				predecessor[id] = node;
				heap.push({-new_dis,id});  
			}
		}
 
		int current_id = net_list[n].second; 
		while(true){  
			int dummy; 
			if(predecessor[current_id] == -1) break;
			int destination = predecessor[current_id];
			answer[n].push_back(make_pair(current_id,destination));
			
			//update weight and demand
			if(current_id == destination + 1){								// left  
				double d = weight_to_demand(weight[current_id][3]);
				d++;
				weight[current_id][3] = demand_to_weight(d); 

				d = weight_to_demand(weight[destination][1]);
				d++;
				weight[destination][1] = demand_to_weight(d);
			}
			else if(current_id == destination - 1){							// right
				double d = weight_to_demand(weight[current_id][1]);
				d++;
				weight[current_id][1] = demand_to_weight(d);

				d = weight_to_demand(weight[destination][3]);
				d++;
				weight[destination][3] = demand_to_weight(d);
			}
			else if(current_id == destination + x_max){						// down
				double d = weight_to_demand(weight[current_id][2]);
				d++;
				weight[current_id][2] = demand_to_weight(d);

				d = weight_to_demand(weight[destination][0]);
				d++;
				weight[destination][0] = demand_to_weight(d);
			}
			else if(current_id == destination - x_max){						// up
				double d = weight_to_demand(weight[current_id][0]);
				d++;
				weight[current_id][0] = demand_to_weight(d);

				d = weight_to_demand(weight[destination][2]);
				d++;
				weight[destination][2] = demand_to_weight(d);
			}
			else{
				cout<<"ERROR!"<<endl;
				cin>>dummy; 
			} 
			current_id = destination;
		} 
 	} 
};

int main(int argc, char *argv[]){
	router dijkstra;  
	dijkstra.read(argv[1]);
	for(int i=0;i<dijkstra.net_num;i++){ 
		dijkstra.route(i);
	}  
	ofstream filePtr;                       
	filePtr.open(argv[2], ios::out);      
 
	for(int i=0;i<dijkstra.net_num;i++){ 
		filePtr<<i<<" "<<dijkstra.answer[i].size()<<endl;
		for(int j=dijkstra.answer[i].size() - 1;j>=0;j--){
 			filePtr<<dijkstra.id_to_cooridate(dijkstra.answer[i][j].second).first;
 			filePtr<<" "<<dijkstra.id_to_cooridate(dijkstra.answer[i][j].second).second;
 			filePtr<<" "<<dijkstra.id_to_cooridate(dijkstra.answer[i][j].first).first;
 			filePtr<<" "<<dijkstra.id_to_cooridate(dijkstra.answer[i][j].first).second<<endl;	
		}
	}

	filePtr.close(); 
}
 