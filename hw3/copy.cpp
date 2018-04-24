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
unordered_map< bitset<50> , int > memory[3][3]; 
 
vector<bitset<50>> generate_next_board(bitset<50> board,char round){
    int a[5][5] , record = 0 , length = 0 , x = 0 , y = 0 , x1 , x2 , y1 , y2; 
    bitset<50> tmp;
    vector<i2tuple> tup;
    vector<bitset<50>> result; 
    for(int i=4;i>=0;i--){ // convert long long to 5*5 array
        for(int j=10;j>=2;j-=2){  
            if(board[(4-i)*10+(10-j)]== true && board[(4-i)*10+(10-j)+1]==false) record = 1;
            else if(board[(4-i)*10+(10-j)]==false && board[(4-i)*10+(10-j)+1]==true) record = 2;
            else record = 0; 
 
            if(record == 0 ) tup.push_back(i2tuple(i,j));
            a[i][(j/2)-1] = record;
        }
    }  
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
    int a[5][5];  
    for(int i=4;i>=0;i--){  
        for(int j=10;j>=2;j-=2){  
            if(board[(4-i)*10+(10-j)]== true && board[(4-i)*10+(10-j)+1]==false) record = 1;
            else if(board[(4-i)*10+(10-j)]==false && board[(4-i)*10+(10-j)+1]==true) record = 2;
            else record = 0;  
            if(record!=0) count++;
            a[i][(j/2)-1] = record;
        }
    } 
    if(count != 22) return 100; 
    else{                       // insert 'X' into the remaining 3 spaces
        for(int i=0;i<5;i++){
            for(int j=0;j<5;j++){
                if(a[i][j]!=1 && a[i][j]!=2) a[i][j] = 1;
            }
        }
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
 
    if(x_score > o_score) return -1; 
    else if(o_score > x_score) return 1; 
    else return 0; 
 
}
int whowin(bitset<50> board,char round,int alpha,int beta){ 
 
    unordered_map<bitset<50>,int>::const_iterator got = memory[alpha+1][beta+1].find(board); 
    if ( got != memory[alpha+1][beta+1].end() ){ 
        return got->second;
    }  
    int answer = evaluate(board);
    if(answer!=100){ 
        memory[alpha+1][beta+1].insert(make_pair(board,answer)); 
        return answer;
    }  
    vector<bitset<50>> next_board = generate_next_board(board,round);
    int tmp = 0; 
    if(round=='O'){ 
        for(int i=0;i<next_board.size();i++){
            tmp = whowin(next_board[i],'X',alpha,beta);
            memory[alpha+1][beta+1].insert(make_pair(next_board[i],tmp));
            alpha = max(alpha,tmp);
            if(beta<=alpha) break;  
        }
 
        return alpha;
    }
    else if(round=='X'){  
        for(int i=0;i<next_board.size();i++){
            tmp = whowin(next_board[i],'O',alpha,beta);
            memory[alpha+1][beta+1].insert(make_pair(next_board[i],tmp));
            beta = min(beta,tmp);
            if(beta<=alpha) break;  
        } 
        return beta;
    }
 
} 
int main(){ 
 
    int num = 0 , result = 0;
    string tmp;
    cin>>num;
    long long int data[num];
    bitset<50> input[num];
    long long int a = 0;  
    for(int i=0;i<num;i++){              // 00: empty 
        a = 0;                           // 01: X   
        for(int j=0;j<5;j++){            // 10: O
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
        input[i] = bitset<50>(data[i]);
    }  
 
    for(int i=0;i<num;i++){
        char next = whonext(input[i]);
        result = whowin(input[i],next,-1,1);
        if(result == 1)cout<<"O win";
        else if(result == -1)cout<<"X win";
        else cout<<"Draw";
        if(i!=num-1) cout<<endl;
    }   
}