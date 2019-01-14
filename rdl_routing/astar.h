#include <bits/stdc++.h>
using namespace std;
 
class node{
public:
	node();
	node(int x , int y);
	int operator[](int x);
	bool operator==(const node& x);
	int manhattan_distance(const node& x);
	void set_f_cost();
	void set_position(int x , int y);
	friend ostream& operator<<(ostream& out, const node& n);
	int pos[2];
	int g_cost , h_cost , f_cost;
	node* parent;
}; 

class net{
public:
	net(string n , node x , node y , int** is_empty);
	string name;
	node source , target; 
	vector<node> node_path;
};

class obstacle{
public:
	obstacle(string n , node a , node b , int** is_empty);
	string name;
	node bottom_left , top_right; 
};
 
struct CmpNodePtrs
{
    bool operator()(const node* lhs, const node* rhs) const;
};

class router{
public:
	router();
	int** is_empty;
 	vector<net> net_list;
 	vector<obstacle> obstacle_list;
 	int x_max , y_max;
 	bool read(const string& txt_file);
 	bool routing(int h);
}; 
 