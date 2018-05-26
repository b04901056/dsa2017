#include<iostream>
#include<deque>
#include<string.h>
#include<unordered_map>
#include <string>
using namespace std;

const int x = 29;
const int m = 1000000007;
const int th = 20; 

unordered_map <long long int,int> roll_hash;
deque <char> s;
char ss[100001],ask[100001];   
 
void update_front(char n){
    long long int pre = 0; 
    s.push_front(n);
    for(int i=0; i<s.size() ;i++){
        pre = ((pre * x)% m + s[i]) % m; 
        roll_hash[pre] += 1;
        if(i==th-1) break;
    } 
    /*for (int i=0;i<s.size();i++) printf("%c ",s[i]);
    for ( auto it = roll_hash.begin(); it != roll_hash.end(); ++it ) cout << " " << it->first << ":" << it->second;
    cout<<endl;*/
}

void update_back(char n){ 
    long long int pre = 0 , count = 0 , tmp = 1;
    s.push_back(n);
    for(int i=s.size()-1; i>=0 ; i--){
        if( count > 0 ) tmp = (tmp * x) % m;  
        pre = (pre + (s[i] * tmp) % m ) % m;  
        //pre = (pre + (s[i] * p[count]) % m ) % m; 
        roll_hash[pre] += 1; 
        if(count==th-1) break;
        count++;
    } 
    /*for(int i=0;i<s.size();i++)  printf("%c ",s[i]); 
    for ( auto it = roll_hash.begin(); it != roll_hash.end(); ++it ) cout << " " << it->first << ":" << it->second ; 
    cout<<endl;*/
} 

int main(){  
	int q=0,cmd=0,number=0;
    char add[1];
    string str;
    scanf("%s",ss);  
    int len = strlen(ss);
    for(int i=len-1;i>=0;i--){
        update_front(ss[i]);  
    }  
    scanf("%d",&q); 
 
	for(int k=0;k<q;k++){
        scanf("%d",&cmd);
		if(cmd==1){
            scanf("%s",add);
			update_front(add[0]);
            //str = add[0] + str ;
		}
		else if(cmd==2){
            scanf("%s",add);
			update_back(add[0]);
            //str = str + add[0] ;
		}
		else if(cmd==3){ 
            scanf("%s",ask); 
            len = strlen(ask); 
            int size = int(s.size());  
            if( len > size ) number = 0; 
            else if( len > th ){ 
                long long int a = 0 , b = 0 , tmp = 1 ;  
                for(int i=0;i<len;i++){
                    b = ((b * x)%m + ask[i]) % m; 
                    a = ((a * x)%m + s[i]) % m;
                    if(i!=0) tmp = (tmp * x) % m ;
                } 
                if( a == b ) number++; 
                for(int i=len;i<size;i++){ 
                    a -= ( s[i-len] * tmp ) % m ; 
                    if(a<0) a += m ;
                    a = (( a * x ) % m + s[i] ) % m ;
                    if( a == b ) number++; 
                }
            }
            else{ 
                long long int pre = 0;
                for(int i=0;i<len;i++){
                    pre = ((pre * x)%m + ask[i]) % m; 
                }
                number = roll_hash[pre];
            }
            if(k==q-1) printf("%d",number);
            else printf("%d\n",number);
            number = 0; 
		}  
    }         
}
