#include <iostream>
#include <unordered_map>
#include <math>
#include <vector>
#include <string>
using namespace std;

// 0: empty
// 1: X 
// 2: O

// 0:  draw
// -1: X wins 
// 1:  O wins

vector<long long int> generate_next_board(long long int){

}

int evaluate(long long int board){
	int o_score = 0,x_score = 0,check_col=0,check_row=0,check_l_oblique=0,check_r_oblique=0,record = 0;
	int a[5][5];
	for(int i=4;i>=0;i--){ // convert long long to 5*5 array
		for(int j=4;j>=0;j--){
			record = board % 4;
			cout<<"record: "<<record<<endl;
			board /= 4 ;
			a[i][j] = record;
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
		check_l_oblique = 0;
		check_r_oblique = 0;
		if(a[i][i]==1) check_l_oblique-=1;
		else if(a[i][i]==2) check_l_oblique+=1;
		if(a[i][i]==1) check_r_oblique-=1;
		else if(a[i][i]==2) check_r_oblique+=1;
	}
	if(check_l_oblique>=3) o_score++;
	else if(check_l_oblique<=-3) x_score++;
	if(check_r_oblique>=3) o_score++;
	else if(check_r_oblique<=-3) x_score++;

	if(x_score > o_score) return -1;
	else if(x_score < o_score) return 1;
	else return 0;
}

int whowin(long long int board,char round){

    unordered_map<long long int, int>::const_iterator got = map.find (board);
    if ( got == map.end() ) cout << "not found"; 
    else{
    	cout << got->first << " is " << got->second; 
    	return got->second;
    }

	int result = 0 , nextresult = 0;
	vector<long long int> next_board = generate_next_board(board);
	if(board>=pow(2,48)) return evaluate(board);
	if(round=='X'){
		result = 1;
		for (int i = 0;i<next_board.size();i++){
			nextresult = whowin(next_board[i], 'O');
	 		if (nextresult == -1) result = -1; 
	 		else if (nextresult == 0 && result == 1) result = 0; 
    	}
    	return result;
	}
	else if(round=='O'){
		result = -1;
		for (int i = 0;i<next_board.size();i++){
			nextresult = whowin(next_board[i], 'X');
	 		if (nextresult == 1) result = 1; 
	 		else if (nextresult == 0 && result == -1) result = 0; 
    	}
    	return result;
	}
}


int main(){
	unordered_map<long long int, int> map;
	long long int a = 0;
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
    unordered_map<long long int, int>::const_iterator got = map.find (682);
    if ( got == map.end() ){
    	cout << "not found";
    }
    else{
    	std::cout << got->first << " is " << got->second;
    }	

    cout<<"read in"<<endl;
    cout<< evaluate(got->first+1+16)<<endl;

}