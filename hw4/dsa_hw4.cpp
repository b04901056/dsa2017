#include<iostream>
#include<deque>
#include<string.h>
using namespace std;

int main(){
    deque<char> s;
		int q,cmd;
		char ss[100000],add;
		scanf("%s",ss); 
    scanf("%d",&q);

    for(int i=0;i<strlen(ss);i++){
		    s.push_back(ss[i]);
		}		/*		
    for(int i=0;i<s.size();i++){
        printf("%c",s[i]);
		}                               
    cout<<endl<<"q="<<q<<endl;*/
		for(int k=0;k<q;k++){
				printf("k= %d",k);
        scanf("%d",&cmd);
				if(cmd==1){
						printf("mode 1\n");
            scanf("%s",add);
						s.push_front(add);
				}
				else if(cmd==2){
					  printf("mode 2\n");
            scanf("%s",add);
						s.push_back(add);
				}
				else if(cmd==3){
						printf("mode 3\n");
			  }
				else {
				    printf("wrong command\n");
				}
		}
		return 0;
}
