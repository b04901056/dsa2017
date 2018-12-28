#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]){
	ofstream filePtr;                       
	filePtr.open(argv[1], ios::out);      
 	
 	filePtr<<"grid "<<argv[2]<<" "<<argv[2]<<endl;
 	filePtr<<"capacity "<<argv[3]<<endl;
 	filePtr<<"num net "<<argv[4]<<endl;
 	int max = atoi(argv[2]) - 1; 
 	int net_num = atoi(argv[4]);  

	for(int i=0;i<net_num;i++){ 
		filePtr<<i<<" "; 
		int a = rand() % (max + 1);
		int b = rand() % (max + 1);
		int c = rand() % (max + 1);
		int d = rand() % (max + 1);
		while(c == a && d == b){
			c = rand() % (max + 1);
			d = rand() % (max + 1);
		}
		filePtr<<a;
		filePtr<<" "<<b;
		filePtr<<" "<<c;
		filePtr<<" "<<d<<endl;	 
	}
}