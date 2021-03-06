#include <iostream>
#include <math.h>
using namespace std;
  
bool can_be_parabola(long long **arr ,int j,int k){
	long long delta_a , delta , delta_b; 
	delta_a = (arr[j][1]*arr[k][0])-(arr[k][1]*arr[j][0]);
	delta_b = (arr[j][0]*arr[j][0]*arr[k][1])-(arr[k][0]*arr[k][0]*arr[j][1]);
	delta = (arr[j][0]*arr[j][0]*arr[k][0])-(arr[j][0]*arr[k][0]*arr[k][0]);
	if( delta_a * delta < 0  && delta_b * delta > 0 ) return true;
	else return false; 
}

bool is_on_parabola(long long **arr ,int j,int k,int l){ 
	long long delta_a1 ,delta_b1 , delta1;
	//long long delta_a2 ,delta_b2 , delta2;
	//long double a1 , a2 , b1 ,b2;
	delta_a1 = (arr[j][1]*arr[k][0])-(arr[k][1]*arr[j][0]);
	delta_b1 = (arr[j][0]*arr[j][0]*arr[k][1])-(arr[k][0]*arr[k][0]*arr[j][1]);
	delta1 = (arr[j][0]*arr[j][0]*arr[k][0])-(arr[j][0]*arr[k][0]*arr[k][0]);

	if( delta_a1 * arr[l][0] * arr[l][0] + delta_b1 * arr[l][0] ==  arr[l][1] * delta1) return true;
	else return false;
	/*
	delta_a2 = (arr[j][1]*arr[l][0])-(arr[l][1]*arr[j][0]);
	delta_b2 = (arr[j][0]*arr[j][0]*arr[l][1])-(arr[l][0]*arr[l][0]*arr[j][1]);
	delta2 = (arr[j][0]*arr[j][0]*arr[l][0])-(arr[j][0]*arr[l][0]*arr[l][0]);
 
	delta1 =  double(delta1);
	delta_a1 =  double(delta_a1);
	delta_b1 =  double(delta_b1);
	delta2 =  double(delta2);
	delta_a2 =  double(delta_a2);
	delta_b2 =  double(delta_b2);

	if( delta1 * delta2 == 0 ) return false; 
	a1 =  delta_a1 /  delta1;
	a2 =  delta_a2 /  delta2;
	b1 =  delta_b1 /  delta1;
	b2 =  delta_b2 /  delta2;

	if( fabs( a1 - a2 ) < eps && fabs( b1 - b2 ) < eps) return true;
	else return false; */
}

int dp[33554432];
 
int main(){
	int T,n;
	long long tmp;
	scanf("%d",&T);
	for(int q=0;q<T;q++){
		scanf("%d",&n);
		long long** a;
		a = new long long* [n]; 
		for(int i=0;i<n;i++) a[i] = new long long[2];
		int all_num = 1 << n; 
		  
		for(int i=0;i<all_num;i++){
			int first = i , count = 0 ;
			while(first!=0){
				if(first%2==1) count ++;
				first = first >> 1 ; 
			} 
			dp[i] = count;
		}
 
		// read in
		for(int i=0;i<n;i++){
			scanf("%lld",&tmp);
			a[i][0] = tmp;
			scanf("%lld",&tmp);
			a[i][1] = tmp;
		} 
		  
		cout<<"preprocess ..."<<endl;
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				//cout<<i<<" "<<j<<endl;
				rec[i][j] = 0;
				if(i==j) continue;
				long long int p = (1 << i )+ (1 << j) ;
				for(int t=0;t<n;t++){ 
					cout<<"i="<<i<<" "<<"j="<<j<<" "<<"t="<<t<<" "<<endl;
					if( t==i || t==j ) continue;
					if( ( a[i][1] / a[i][0] ) == ( a[j][1] / a[j][0] ) ){
						p = 0;
						break;
					}
					if( a[i][0] == a[j][0] && a[i][1] != a[j][1]){
						p = 0;
						break;
					} 
					if(is_on_parabola(a,i,j,t)){
						p += 1 << t ;
						cout<<"get"<<endl;
					} 
				}
				rec[i][j] = p;
			}
		} 
		/*
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				cout<<rec[i][j]<<" ";
			}
			cout<<endl;
		}*/


		for(int i=1;i<all_num;i++){ 
			for(int j=0;j<n;j++){ 
				if( ((1 << j) & i) == 0 ) continue;
				for(int k=0;k<n;k++){  
					if( ((1 << k) & i) == 0 ) continue;
					if( j == k ) {
						dp[i] = min ( dp[i] , dp[i-(( 1 << j )) ] + 1 );
					}   
					if(!can_be_parabola(a,j,k)) continue;
					
					int p = ( 1 << j ) + ( 1 << k ); 
					for(int t=k+1;t<n;t++){ 
						if( ((1 << t) & i) == 0 ) continue;
						if( t == j || t == k ) continue;
						if(is_on_parabola(a,j,k,t)){ 
							p += ( 1 << t );
						}
					}
					//cout<<"p="<<p<<endl;
					//cout<<dp[i]<<" "<<dp[i-p]+1<<endl;
					dp[i] = min(dp[i],dp[i-p]+1);  
				}
				break;
			}
		}
		cout<<dp[all_num-1];
		if(q<T-1) cout<<endl;
		//return 0;
		for(int i=0;i<n;i++) delete[] a[i];
		delete a; 	 
	}
}