#include <iostream>
using namespace std;

struct coordinate{
	int x;
	int y; 
};

bool can_be_parabola(int **arr ,int j,int k){
	long long delta_a , delta; 
	delta_a = (arr[j][1]*arr[k][0])-(arr[k][1]*arr[j][0]);
	//delta_y1 = (arr[j][0]*arr[j][0]*arr[k][1])-(arr[k][0]*arr[k][0]*arr[j][1]);
	delta = (arr[j][0]*arr[j][0]*arr[k][0])-(arr[j][0]*arr[k][0]*arr[k][0]);
	if( delta_a * delta < 0 ) return true;
	else return false; 
}

bool is_on_parabola(int **arr ,int j,int k,int l){
	long long delta_a1 ,delta_b1 , delta1;
	long long delta_a2 ,delta_b2 , delta2;
	long double a1 , a2 , b1 ,b2;
	delta_a1 = (arr[j][1]*arr[k][0])-(arr[k][1]*arr[j][0]);
	delta_b1 = (arr[j][0]*arr[j][0]*arr[k][1])-(arr[k][0]*arr[k][0]*arr[j][1]);
	delta1 = (arr[j][0]*arr[j][0]*arr[k][0])-(arr[j][0]*arr[k][0]*arr[k][0]);
	delta_a2 = (arr[j][1]*arr[l][0])-(arr[l][1]*arr[j][0]);
	delta_b2 = (arr[j][0]*arr[j][0]*arr[l][1])-(arr[l][0]*arr[l][0]*arr[j][1]);
	delta2 = (arr[j][0]*arr[j][0]*arr[l][0])-(arr[j][0]*arr[l][0]*arr[l][0]);

	/*cout<<"delta_a1="<<delta_a1<<endl;
	cout<<"delta_b1="<<delta_b1<<endl;
	cout<<"delta1="<<delta1<<endl;
	cout<<"delta_a2="<<delta_a2<<endl;
	cout<<"delta_b2="<<delta_b2<<endl;
	cout<<"delta2="<<delta2<<endl;*/
	if( delta1 * delta2 == 0 ) return false; 
	a1 = delta_a1 / delta1;
	a2 = delta_a2 / delta2;
	b1 = delta_b1 / delta1;
	b2 = delta_b2 / delta2;

	if( a1 == a2 && b1 == b2 ) return true;
	else return false; 
}

int dp[33554432];
 
int main(){
	int T,n,tmp;
	scanf("%d",&T);
	for(int q=0;q<T;q++){
		scanf("%d",&n);
		int** a;
		a = new int* [n]; 
		for(int i=0;i<n;i++) a[i] = new int[2];
		int all_num = 1 << n; 
	 
		// read in
		for(int i=0;i<n;i++){
			scanf("%d",&tmp);
			a[i][0] = tmp;
			scanf("%d",&tmp);
			a[i][1] = tmp;
		} 
		cout<<is_on_parabola(a,0,1,2)<<endl;
		cout<<can_be_parabola(a,0,1)<<endl;
		cout<<can_be_parabola(a,0,2)<<endl;
		cout<<can_be_parabola(a,2,1)<<endl;
		cout<<can_be_parabola(a,2,4)<<endl;
		cout<<is_on_parabola(a,0,1,3)<<endl;
		cout<<is_on_parabola(a,0,1,4)<<endl;
		cout<<is_on_parabola(a,0,1,5)<<endl;
		cout<<is_on_parabola(a,0,1,6)<<endl;
		return 0;

		 
	}

}