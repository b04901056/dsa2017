#include <iostream>
#include <vector>
#include <deque>
using namespace std; 

int main(){  
	int n,t;
	int inf = 1000000007;
	scanf("%d",&n);
	int a[n+1],arr[n+1];

	for(int i=0;i<n;i++){
		scanf("%d",&t);
		a[i+1]=t;
	}
	for(int i=n;i>=1;i--){
		arr[i] = a[(n+1)-i];
	}
	
 
	int dp[n+1]; 
	int cnt[n+1];
	dp[1] = 1;
	cnt[1] = 1;
	for(int i=2;i<=n;i++){
		dp[i] = 1;
		cnt[i] = 1 ;
		for(int j=1;j<i;j++){
			if(arr[j]<arr[i]) continue;
			if(dp[j]+1 > dp[i]){
				dp[i] = dp[j]+1;
				cnt[i] = cnt[j] % inf;
			}
			else if(dp[j]+1 == dp[i]){
				cnt[i] += cnt[j] % inf;
				cnt[i] = cnt[i] % inf;
			}
		}
 
	}
	int max_dp = 1;
	for(int i=1;i<=n;i++){
		if(dp[i]>max_dp) max_dp = dp[i];
	}
	cout<<max_dp<<endl;
	int num_method = 0; 
	for(int i=1;i<=n;i++){
		if(dp[i]==max_dp) num_method += cnt[i] % inf ;
		num_method = num_method % inf;
	}
	cout<<num_method<<endl;
	vector<int> answer;
	int th = max_dp;
	for(int i=n;i>=1;i--){
		if(dp[i]==th){
			answer.push_back(i);
			th--; 
		} 
	}
	for(int i=0;i<answer.size();i++){
		cout<<(n+1)-answer[i];
		if(i<answer.size()-1) cout<<" ";
	} 
}