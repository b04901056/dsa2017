#include <iostream>
#include <unordered_map>
#include <math.h>
#include <vector>
#include <string>
#include <bitset>
using namespace std;

// 0:  draw
// -1: X wins 
// 1:  O wins
unordered_map<bitset<50>, int> memory;

vector<bitset<50>> generate_next_board(bitset<50> board,char round){
	int a[5][5] , record = 0; 
	unordered_map<int,int> tup;
	vector<bitset<50>> result;
	for(int i=4;i>=0;i--){ // convert long long to 5*5 array
		for(int j=10;j>=2;j-=2){  
			if(board[(4-i)*10+(10-j)]== true && board[(4-i)*10+(10-j)+1]==false) record = 1;
			else if(board[(4-i)*10+(10-j)]==false && board[(4-i)*10+(10-j)+1]==true) record = 2;
			else record = 0; 

			//if(record == 0 ) tup.insert(make_pair(i,j));
			a[i][(j/2)-1] = record;
		}
	}/*
    for(auto it=tup.begin();it!=tup.end();it++){
    	cout<<" "<<it->first<<" : "<<it->second<<endl;
    }*/

}

int evaluate(bitset<50> board){
	int o_score = 0,x_score = 0,check_col=0,check_row=0,check_l_oblique=0,\
	check_r_oblique=0,record = 0,count=0;
	int a[5][5]; 
	for(int i=4;i>=0;i--){ // convert long long to 5*5 array
		for(int j=10;j>=2;j-=2){  
			if(board[(4-i)*10+(10-j)]== true && board[(4-i)*10+(10-j)+1]==false) record = 1;
			else if(board[(4-i)*10+(10-j)]==false && board[(4-i)*10+(10-j)+1]==true) record = 2;
			else record = 0; 
			if(record!=0) count++;
			a[i][(j/2)-1] = record;
		}
	}
	if(count != 22) return 100; // game's not finished yet
	else{                       // insert 'X' into the remaining 3 spaces
		for(int i=0;i<5;i++){
			for(int j=0;j<5;j++){
				if(a[i][j]!=1 && a[i][j]!=2) a[i][j] = 1;
			}
		}
	}
	for(int i=0;i<5;i++){
		for(int j=0;j<5;j++){
			cout<<a[i][j]<<" ";
		}
		cout<<endl;
	}
	for(int i=0;i<5;i++){  // check column & row
		check_col = 0;
		check_row = 0;
		for(int j=0;j<5;j++){
			if(a[i][j]==1) check_row-=1;
			else if(a[i][j]==2) check_row+=1; 

			if(a[j][i]==1) check_col-=1;
			else if(a[j][i]==2) check_col+=1; 
		}
		if(check_col>=3) o_score++;
		else if(check_col<=-3) x_score++;

		if(check_row>=3) o_score++;
		else if(check_row<=-3) x_score++;
	}
	for(int i=0;i<5;i++){ // check oblique
		if(a[i][i]==1) check_l_oblique-=1;
		else if(a[i][i]==2) check_l_oblique+=1;
		if(a[i][4-i]==1) check_r_oblique-=1;
		else if(a[i][4-i]==2) check_r_oblique+=1;
	}
	if(check_l_oblique>=3) o_score++;
	else if(check_l_oblique<=-3) x_score++;
	if(check_r_oblique>=3) o_score++;
	else if(check_r_oblique<=-3) x_score++;
    
    cout<<"x_score = "<<x_score<<endl;
    cout<<"o_score = "<<o_score<<endl;

	if(x_score > o_score) return -1;
	else if(o_score > x_score) return 1;
	else return 0;

}/*
int whowin(bitset<50> board,char round){

    unordered_map<bitset<50>, int>::const_iterator got = memory.find (board);
    if ( got == memory.end() ) cout << "not found"; 
    else{
    	cout << got->first << " is " << got->second; 
    	return got->second;
    } 
	if(evaluate(board)!=100) return evaluate(board);

	int result , nextresult ;
	vector<bitset<50>> next_board = generate_next_board(board,round);

	if(round=='X'){
		result = 1;
		for (int i = 0;i<next_board.size();i++){
			nextresult = whowin(next_board[i], 'O' , memory);
	 		if (nextresult == -1) result = -1; 
	 		else if (nextresult == 0 && result == 1) result = 0; 
    	}
    	return result;
	}
	else if(round=='O'){
		result = -1;
		for (int i = 0;i<next_board.size();i++){
			nextresult = whowin(next_board[i], 'X', memory);
	 		if (nextresult == 1) result = 1; 
	 		else if (nextresult == 0 && result == -1) result = 0; 
    	}
    	return result;
	}
}*/

int main(){
	/*
	bitset<50> a = 0;
    for(int i=0;i<25;i++){
    	cout<<a<<endl;
    	a = a << 1;
    	a++;
    	a = a << 1;
    	map.insert(make_pair(a,1));
    }
    for(auto it=map.begin();it!=map.end();it++){
    	cout<<" "<<it->first<<" : "<<it->second<<endl;
    }
    unordered_map<bitset<50>, int>::const_iterator got = map.find (682);
    if ( got == map.end() ){
    	cout << "not found";
    }
    else{
    	std::cout << got->first << " is " << got->second;
    }	

    cout<<"read in"<<endl;
    cout<< evaluate(got->first+1+16)<<endl;*/
	int num = 0 , result = 0;
	string tmp;
    cin>>num;
    long long int data[num];
    bitset<50> input[50];
    long long int a = 0; 
    cout<<"read in ..."<<endl;
    for(int i=0;i<num;i++){  			// 00: empty 
    	a = 0;               			// 01: X   
		for(int j=0;j<5;j++){			// 10: O
             cin>>tmp;  
             for(int k=0;k<5;k++){
             	a = a << 1;
             	if(tmp[k]=='X'){ 
             		a = a << 1;
             		a++;
             	}
             	else if(tmp[k]=='O'){ 
             		a++;
             		a = a << 1; 
             	}
             	else a = a << 1; 
             }
    	}
    	data[i]=a;
    } 
    for(int i=0;i<num;i++){
    	cout<<i<<":"<<bitset<50>(data[i])<<endl;
    	input[i] = bitset<50>(data[i]);
    } 
    for(int i=0;i<num;i++){
    	cout<<i<<" ";
    	evaluate(input[i]);
    }
    generate_next_board(input[0],'X'); 
    /*
    cout<<"write out ..."<<endl;
	for(int i=0;i<num;i++){
		result = whowin(bitset<50>(input[i]),'O');
		if(result == 1)cout<<"O win"<<endl;
		else if(result == -1)cout<<"X win"<<endl;
		else cout<<"Draw"<<endl;
	}   */
}