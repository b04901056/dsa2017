#include <iostream>
#include <string>
#include <fstream> 
#include <vector> 
#include <algorithm> 
#include <iomanip>  
using namespace std;

class data{
public:
	string date,currency,exchange;
	float max,min;
	long long int cap; 
private:
};

vector<data> set; 
vector<data> set_simple;  

bool myfunction_simple(data const &a,data const &b){
	if(a.date != b.date){
		if(a.date < b.date) return true;
		else return false;
	}
	else if(a.currency != b.currency){
		if(a.currency < b.currency) return true;
		else return false;
	}
	else if(a.exchange != b.exchange){
		if(a.exchange < b.exchange) return true;
		else return false;
	}
}

bool myfunction_cap(data const &a,data const &b){
	if(a.date != b.date){
		if(a.date < b.date) return true;
		else return false;
	}
	else if(a.exchange != b.exchange){
		if(a.exchange < b.exchange) return true;
		else return false;
	}
}

int binary_search_date_left(int left,int right,vector<data>&  a ,string value){   
	if(left == right) {
		if(a[left].date == value) return left;
		else { 
			return -1;
		}
	}
	int mid = left+(right-left)/2; 

	if(value>a[mid].date){  
		return binary_search_date_left(mid + 1,right,a,value);
	}
	else {  
		return binary_search_date_left(left,mid ,a,value);
	}   

} 

int binary_search_currency_left(int left,int right,vector<data>&  a ,string value){ 
	if(left == right) {
		if(a[left].currency == value) return left;
		else { 
			return -1;
		}
	}
	int mid = left+(right-left)/2;  
	if(value>a[mid].currency){  
		return binary_search_currency_left(mid + 1,right,a,value);
	}
	else {  
		return binary_search_currency_left(left,mid ,a,value);
	}   
} 

int binary_search_exchange_left(int left,int right,vector<data>&  a ,string value){ 
	if(left == right) {
		if(a[left].exchange == value) return left;
		else { 
			return -1;
		}
	}
	int mid = left+(right-left)/2; 
	if(value>a[mid].exchange){  
		return binary_search_exchange_left(mid + 1,right,a,value);
	}
	else {  
		return binary_search_exchange_left(left,mid ,a,value); 
	}   
} 

int binary_search_date_right(int left,int right,vector<data>&  a ,string value){  
	if(left == right) {
		if(a[left].date == value) return left;
		else { 
			return -1;
		}
	}
	int mid = left+(right-left+1)/2; 
	if(value<a[mid].date){  
		return binary_search_date_right(left,mid-1,a,value);
	}
	else {  
		return binary_search_date_right(mid,right ,a,value);
	}   

} 

int binary_search_currency_right(int left,int right,vector<data>&  a ,string value){  
	if(left == right) {
		if(a[left].currency == value) return left;
		else { 
			return -1;
		}
	}
	int mid = left+(right-left+1)/2; 
	if(value<a[mid].currency){  
		return binary_search_currency_right(left,mid-1,a,value);
	}
	else {  
		return binary_search_currency_right(mid,right ,a,value);
	}   

} 
int binary_search_exchange_right(int left,int right,vector<data>&  a ,string value){ 
	if(left == right) {
		if(a[left].exchange == value) return left;
		else { 
			return -1;
		}
	}
	int mid = left+(right-left+1)/2;

	if(value<a[mid].exchange){  
		return binary_search_exchange_right(left,mid-1,a,value);
	}
	else {  
		return binary_search_exchange_right(mid,right ,a,value);
	}   

} 

int main(int argc,char** argv){ 
	fstream file , result;      
	string str,instrution; 
	file.open(argv[1], ios::in);   
	while(getline(file,str,'\n')){
		size_t pos = 0;
		data in; 
		string delimiter = "\t"; 
		string token;

		pos = str.find(delimiter);
		token = str.substr(0, pos);
		in.date=token; 
		str.erase(0, pos + delimiter.length());

		pos = str.find(delimiter);
		token = str.substr(0, pos);
		in.currency=token; 
		str.erase(0, pos + delimiter.length());

		pos = str.find(delimiter);
		token = str.substr(0, pos);
		in.exchange=token; 
		str.erase(0, pos + delimiter.length());

		pos = str.find(delimiter);
		token = str.substr(0, pos);
		in.min=stof(token);   
		str.erase(0, pos + delimiter.length());

		pos = str.find(delimiter);
		token = str.substr(0, pos);
		in.max=stof(token);   
		str.erase(0, pos + delimiter.length());

		pos = str.find(delimiter); 
		token = str.substr(0, pos); 
		in.cap=stoll(token);  

		set.push_back(in);
	} 
	file.close(); 
	sort (set.begin(), set.end(), myfunction_simple);
	set_simple = set;
	sort (set.begin(), set.end(), myfunction_cap); 

	//file.open(argv[2], ios::in); 
	//result.open(argv[3],ios::out); 
	size_t pos = 0; 
	string delimiter ="\t"; 
	string token; 
	int left=0;
	int right=set.size()-1; 
	string query_date,query_currency,query_exchange,price_minmax,price_date,price_currency,cap_date,cap_exchange; 
	 
	while(getline(cin,str,'\n')){ 
		pos = 0 ;  
		left = 0 ;
		right=set.size()-1; 

		pos = str.find(delimiter);
		token = str.substr(0, pos); 
		str.erase(0, pos + delimiter.length());

		if(token=="query"){
			pos = str.find(delimiter);
			token = str.substr(0, pos); 
			str.erase(0, pos + delimiter.length());
			query_date=token;

			pos = str.find(delimiter);
			token = str.substr(0, pos);    
			str.erase(0, pos + delimiter.length());
			query_currency=token;

			pos = str.find(delimiter);
			token = str.substr(0, pos); 
			str.erase(0, pos + delimiter.length());
			query_exchange=token;  

			left = binary_search_date_left(left,right,set_simple,query_date); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			} 
			right = binary_search_date_right(left,right,set_simple,query_date); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			} 
			left = binary_search_currency_left(left,right,set_simple,query_currency); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			} 
			right = binary_search_currency_right(left,right,set_simple,query_currency); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			} 
			left = binary_search_exchange_left(left,right,set_simple,query_exchange); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			} 
			right = binary_search_exchange_right(left,right,set_simple,query_exchange); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			} 
			cout << fixed << setprecision(4)<<set_simple[left].min<<" "<<set_simple[left].max<<" "<<set_simple[left].cap<<'\n';
		}
		else if(token=="price"){  
			pos = str.find(delimiter);
			token = str.substr(0, pos); 
			str.erase(0, pos + delimiter.length());
			price_minmax=token;
	 
			pos = str.find(delimiter);
			token = str.substr(0, pos);    
			str.erase(0, pos + delimiter.length());
			price_date=token;

			pos = str.find(delimiter);
			token = str.substr(0, pos); 
			str.erase(0, pos + delimiter.length());
			price_currency=token; 
 
			
			left = binary_search_date_left(left,right,set_simple,price_date); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			} 
			right = binary_search_date_right(left,right,set_simple,price_date); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			} 
			left = binary_search_currency_left(left,right,set_simple,price_currency); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			} 
			right = binary_search_currency_right(left,right,set_simple,price_currency); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			}

			float candidate=0;
			if(price_minmax=="max"){
				for(int i=left;i<right+1;i++){
					if(set_simple[i].max>candidate) candidate = set_simple[i].max;
				}
			}
			else{
				candidate = 99999999999;
				for(int i=left;i<right+1;i++){
					if(set_simple[i].min<candidate) candidate = set_simple[i].min;
				} 
			}

			cout<< fixed << setprecision(4)<<candidate<<'\n';
		}
		else if(token=="cap"){
			pos = str.find(delimiter);
			token = str.substr(0, pos); 
			str.erase(0, pos + delimiter.length());
			cap_date=token;
	 
			pos = str.find(delimiter);
			token = str.substr(0, pos);    
			str.erase(0, pos + delimiter.length());
			cap_exchange=token;  
			
			left = binary_search_date_left(left,right,set,cap_date); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			} 
			right = binary_search_date_right(left,right,set,cap_date); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			} 
			left = binary_search_exchange_left(left,right,set,cap_exchange); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			} 
			right = binary_search_exchange_right(left,right,set,cap_exchange); 
			if(left==-1 || right ==-1){
				cout<<"none"<<endl;
				continue;
			}

			long long int total = 0 ; 
			for(int i=left;i<right+1;i++) {
				total+=set[i].cap;
			}

			cout<<total<<'\n';
		}
		else break; 
	} 
	return 0;
}