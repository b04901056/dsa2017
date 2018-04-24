#include <iostream>
#include <unordered_map>
#include <math.h>
#include <vector>
#include <string>
#include <tuple>
#include <bitset>
using namespace std;

// 0:  draw
// -1: X wins 
// 1:  O wins

typedef tuple<int,int> i2tuple; 
typedef tuple<int,int> mapout;
 

class mapin{
public:
	mapin(bitset<50> board_,int alpha_,int beta_){
		board = board_;
		alpha = alpha_;
		beta = beta_;
	} 
	bool operator== (const mapin& p) const
    {
        return board==p.board && alpha==p.alpha && beta==p.beta;
    }
	bitset<50> board;
	int alpha;
	int beta;
};

namespace std {

  template <>
  struct hash<mapin>
  {
    std::size_t operator()(const mapin& k) const
    {
      using std::size_t;
      using std::hash;
      using std::string; 

      return ((hash<std::bitset<50>>()(k.board)
               ^ (hash<int>()(k.alpha) << 1)) >> 1)
               ^ (hash<int>()(k.beta) << 1);
    }
  };

}

unordered_map<mapin, mapout> memory;

int** convert2array(bitset<50> board){ // convert long long to 5*5 array
	int **a , record = 0 ;
	a = new int*[5];
	for(int i=0;i<5;i++) a[i] = new int [5]; 
	for(int k=0;k<5;k++){
		for(int y=0;y<5;y++) a[k][y]=0; 
	}
	for(int i=4;i>=0;i--){  
		for(int j=10;j>=2;j-=2){  
			if(board[(4-i)*10+(10-j)]== true && board[(4-i)*10+(10-j)+1]==false) record = 1;
			else if(board[(4-i)*10+(10-j)]==false && board[(4-i)*10+(10-j)+1]==true) record = 2;
			else record = 0;  
			a[i][(j/2)-1] = record;
		}
	}
	return a;
}

vector<bitset<50>> generate_next_board(bitset<50> board,char round){
	int **a , record = 0 , length = 0 , x = 0 , y = 0 , x1 , x2 , y1 , y2; 
	bitset<50> tmp;
	vector<i2tuple> tup;
	vector<bitset<50>> result;
	cout<<"generate_next_board"<<endl<<endl;
	//cout<<board<<endl;
	a = new int*[5];
	for(int i=0;i<5;i++) a[i] = new int [5]; 
	for(int i=4;i>=0;i--){ // convert long long to 5*5 array
		for(int j=10;j>=2;j-=2){  
			if(board[(4-i)*10+(10-j)]== true && board[(4-i)*10+(10-j)+1]==false) record = 1;
			else if(board[(4-i)*10+(10-j)]==false && board[(4-i)*10+(10-j)+1]==true) record = 2;
			else record = 0; 

			if(record == 0 ) tup.push_back(i2tuple(i,j));
			a[i][(j/2)-1] = record;
		}
	}

	cout<<"original_board:"<<endl;
	for(int i=0;i<5;i++){
		for(int j=0;j<5;j++){
			cout<<a[i][j]<<" ";
		}
		cout<<endl;
	}
	for(int i=0;i<tup.size();i++){
		x1 = get<0>(tup[i]);
		y1 = get<1>(tup[i]); 
		cout<<x1<<" "<<(y1/2)-1<<" "<<endl;
	}
	cout<<endl;
	for(int i=0;i<tup.size();i++){
		for(int j=i+1;j<tup.size();j++){
			tmp = board;
			x1 = get<0>(tup[i]);
			y1 = get<1>(tup[i]);
			x2 = get<0>(tup[j]);
			y2 = get<1>(tup[j]);
            if(round=='X'){ 
            	tmp[(4-x1)*10+(10-y1)] = true;
            	tmp[(4-x1)*10+(10-y1)+1] = false;
            	tmp[(4-x2)*10+(10-y2)] = true;
            	tmp[(4-x2)*10+(10-y2)+1] = false;
            }
            else if(round=='O'){
            	tmp[(4-x1)*10+(10-y1)] = false;
            	tmp[(4-x1)*10+(10-y1)+1] = true;
            	tmp[(4-x2)*10+(10-y2)] = false;
            	tmp[(4-x2)*10+(10-y2)+1] = true;
            } 
			result.push_back(tmp);
		}
	}
	
	cout<<"next:"<<endl;
	for(int z=0;z<result.size();z++){ 
		a = convert2array(result[z]); 
		for(int i=0;i<5;i++){
			for(int j=0;j<5;j++){
				cout<<a[i][j]<<" ";
			}
			cout<<endl;
		}
		cout<<endl;
	}
	cout<<endl;
	for(int i=0;i<5;i++) delete a[i]; 
	delete a;
	return result;
}

char whonext(bitset<50> board){
	int x = 0 , o = 0; 
	for(int i=4;i>=0;i--){  
		for(int j=10;j>=2;j-=2){  
			if(board[(4-i)*10+(10-j)]== true && board[(4-i)*10+(10-j)+1]==false) x++;
			else if(board[(4-i)*10+(10-j)]==false && board[(4-i)*10+(10-j)+1]==true) o++; 
		}
	}
	if( o == x + 2 ) return 'X';
	else if( o == x ) return 'O'; 
}

int evaluate(bitset<50> board){
	int o_score = 0,x_score = 0,check_col=0,check_row=0,check_l_oblique=0,\
	check_r_oblique=0,record = 0,count=0;
	int **a; 
	cout<<"evaluate"<<endl;
	a = new int*[5];
	for(int i=0;i<5;i++) a[i] = new int [5]; 
	a = convert2array(board); 
	for(int i=0;i<5;i++){
		for(int j=0;j<5;j++){
			if(a[i][j]!=0) count++;
		}
	}
	if(count != 22) {   // game's not finished yet
		cout<<"game's not finished yet"<<endl<<endl;
		return 100;
	}
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
    cout<<"o_score = "<<o_score<<endl<<endl;

	for(int i=0;i<5;i++) delete a[i]; 
	delete a;

	if(x_score > o_score) {
		cout<<"X WIN"<<endl<<endl;
		return -1;
	} 
	else if(o_score > x_score){
		cout<<"O WIN"<<endl<<endl; 
		return 1;
	}
	else{
		cout<<"DRAW"<<endl<<endl;
		return 0;		
	}

}
int whowin(bitset<50> board,char round,int alpha,int beta){
	int **a; 
	a = new int*[5];
	for(int i=0;i<5;i++) a[i] = new int [5]; 
	a = convert2array(board);

    cout<<"whowin,current board : "<<round<<"'s turn"<<endl; 
	for(int i=0;i<5;i++){
		for(int j=0;j<5;j++){
			cout<<a[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
    unordered_map<mapin, mapout>::const_iterator got = memory.find (mapin(board,alpha,beta));
    if ( got != memory.end() ){
    	//cout << got->first << " is " << got->second;
    	cout<< "find answer in memory"<<endl; 
    	return get<0>(got->second);
    }  
    int answer = evaluate(board);
	if(answer!=100){ 
		memory.insert(make_pair(mapin(board,alpha,beta),mapout(answer,answer)));
		cout<< "reach bottom"<<endl; 
		return answer;
	}  
	vector<bitset<50>> next_board = generate_next_board(board,round);
	int tmp = 0;
	/*
	int result , nextresult ;

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
	}*/
	if(round=='O'){
		cout<< "O 's turn"<<endl; 
		for(int i=0;i<next_board.size();i++){
			tmp = whowin(next_board[i],'X',alpha,beta);
			memory.insert(make_pair(mapin(next_board[i],alpha,beta),mapout(alpha,tmp)));
			alpha = max(alpha,tmp);
			if(beta<=alpha) break; 
			cout<<"alpha = "<<alpha<<endl;
		    cout<<"beta = "<<beta<<endl<<endl;
		}

		return alpha;
	}
	else if(round=='X'){ 
		cout<< "X 's turn"<<endl; 
		for(int i=0;i<next_board.size();i++){
			tmp = whowin(next_board[i],'O',alpha,beta);
			memory.insert(make_pair(mapin(next_board[i],alpha,beta),mapout(tmp,beta)));
			beta = min(beta,tmp);
			if(beta<=alpha) break; 
			cout<<"alpha = "<<alpha<<endl;
		    cout<<"beta = "<<beta<<endl<<endl;
		} 
		return beta;
	}

}

int main(){ 

	int num = 0 , result = 0;
	string tmp;
    cin>>num;
    long long int data[num];
    bitset<50> input[50];
    long long int a = 0; 
    cout<<"read in ..."<<endl<<endl;
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
    	//cout<<i<<":"<<bitset<50>(data[i])<<endl;
    	input[i] = bitset<50>(data[i]);
    } 
    /*
    for(int i=0;i<num;i++){
    	cout<<"i="<<i<<" "<<endl;
    	evaluate(input[i]);
    } 
    for(int i=0;i<num;i++){
    	cout<<"i= "<<i<<"  "<<"next_board:"<<endl;
    	generate_next_board(input[i],'X'); 
    }*/
    
	char next = whonext(input[3]);
	result = whowin(input[3],next,-1,1);
	if(result == 1)cout<<"Final O win"<<endl;
	else if(result == -1)cout<<"Final X win"<<endl;
	else cout<<"Final Draw"<<endl;
    /*
    cout<<"write out ..."<<endl;
	for(int i=0;i<num;i++){
		char next = whonext(input[i]);
		result = whowin(input[i],next,1,-1);
		if(result == 1)cout<<"O win"<<endl;
		else if(result == -1)cout<<"X win"<<endl;
		else cout<<"Draw"<<endl;
	}   */
}