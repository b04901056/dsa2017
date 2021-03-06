#include<iostream>
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
	cout<<"binary_search_date_left"<<endl;
	cout<<"left="<<left<<" "<<"right="<<right<<" "<<endl;
	cout<<"("<<a[left].date<<") "<<a[right].date<<endl;
	cout<<"("<<value<<")"<<endl;
	if(left == right) {
		if(a[left].date == value) return left;
		else { 
			return -1;
		}
	}
	int mid = left+(right-left)/2;
	cout<<mid<<endl;

	if(value>a[mid].date){  
		return binary_search_date_left(mid + 1,right,a,value);
	}
	else {  
		return binary_search_date_left(left,mid ,a,value);
	}   

} 

int binary_search_currency_left(int left,int right,vector<data>&  a ,string value){
	cout<<"binary_search_currency_left"<<endl;
	cout<<"left="<<left<<" "<<"right="<<right<<" "<<endl;
	cout<<"("<<a[left].currency<<")("<<a[right].currency<<")"<<endl;
	cout<<"("<<value<<")"<<endl;
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
	cout<<"binary_search_exchange_left"<<endl;
	cout<<"left="<<left<<" "<<"right="<<right<<" "<<endl;
	cout<<"("<<a[left].exchange<<")("<<a[right].exchange<<")"<<endl;
	cout<<"("<<value<<")"<<endl;
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
	cout<<"binary_search_date_right"<<endl;
	cout<<"left="<<left<<" "<<"right="<<right<<" "<<endl;
	cout<<"("<<a[left].date<<") "<<a[right].date<<endl;
	cout<<"("<<value<<")"<<endl;
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
	cout<<"binary_search_currency_right"<<endl;
	cout<<"left="<<left<<" "<<"right="<<right<<" "<<endl;
	cout<<"("<<a[left].currency<<")("<<a[right].currency<<")"<<endl;
	cout<<"("<<value<<")"<<endl;
	if(left == right) {
		if(a[left].currency == value) return left;
		else { 
			return -1;
		}
	}
	int mid = left+(right-left+1)/2;
	//cout<<left<<" "<<right<<" "<<mid<<endl;
	if(value<a[mid].currency){  
		return binary_search_currency_right(left,mid-1,a,value);
	}
	else {  
		return binary_search_currency_right(mid,right ,a,value);
	}   

} 
int binary_search_exchange_right(int left,int right,vector<data>&  a ,string value){ 
	cout<<"binary_search_exchange_right"<<endl;
	cout<<"left="<<left<<" "<<"right="<<right<<" "<<endl;
	cout<<"("<<a[left].exchange<<")("<<a[right].exchange<<")"<<endl;
	cout<<"("<<value<<")"<<endl;
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
	cout<<"read in ..."<<endl;  
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

	// simple /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
/*
	cout<<"sort ..."<<set.size()<<endl;  
	sort (set.begin(), set.end(), myfunction_simple);
		result.open("myfunction_simple.txt",ios::out);
	for(int i=0;i<set.size();i++){
		result<<set[i].date<<"\t"<<set[i].currency<<"\t"<<set[i].exchange<<"\t"<<set[i].max<<"\t"<<set[i].min<<"\t"<<set[i].cap<<'\n';
	}   
	result.close();
	string query_date,query_currency,query_exchange; 
	file.open(argv[2], ios::in); 
	result.open("result_1.txt",ios::out); 
	while(getline(file,str,'\n')){ 
		if(str=="end") break; 
		size_t pos = 0; 
		string delimiter ="\t"; 
		string token; 
		int left=0;
		int right=set.size()-1; 
 
		pos = str.find(delimiter);
		token = str.substr(0, pos); 
		str.erase(0, pos + delimiter.length());

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
		cout<<query_date<<" "<<query_currency<<" "<<query_exchange<<endl;
		left = binary_search_date_left(left,right,set,query_date); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}
		cout<<"return_date_left "<<left<<" "<<right<<endl;
		right = binary_search_date_right(left,right,set,query_date); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}
		cout<<"return_date_right "<<left<<" "<<right<<endl;
		left = binary_search_currency_left(left,right,set,query_currency); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}
		cout<<"return_cur_left "<<left<<" "<<right<<endl;
		right = binary_search_currency_right(left,right,set,query_currency); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}
		cout<<"return_cur_right "<<left<<" "<<right<<endl;
		left = binary_search_exchange_left(left,right,set,query_exchange); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}
		cout<<"return_ex_left "<<left<<" "<<right<<endl;
		right = binary_search_exchange_right(left,right,set,query_exchange); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}
		cout<<"return_ex_right "<<left<<" "<<right<<endl;
		result<< fixed << setprecision(4)<<set[left].max<<" "<<set[left].min<<" "<<set[left].cap<<'\n';
	} */
	 
	
	// daily /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
	/*cout<<"sort ..."<<set.size()<<endl;  
	sort (set.begin(), set.end(), myfunction_simple); 

	string price_minmax,price_date,price_currency; 
	file.open(argv[2], ios::in); 
	result.open("result_2.txt",ios::out); 
	while(getline(file,str,'\n')){ 
		if(str=="end") break; 
		size_t pos = 0; 
		string delimiter ="\t"; 
		string token; 
		int left=0;
		int right=set.size()-1; 
 
		pos = str.find(delimiter);
		token = str.substr(0, pos); 
		str.erase(0, pos + delimiter.length());

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

		cout<<price_minmax<<" "<<price_date<<" "<<price_currency<<endl<<endl;
		
		left = binary_search_date_left(left,right,set,price_date); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}
		cout<<"return_date_left "<<left<<" "<<right<<endl;
		right = binary_search_date_right(left,right,set,price_date); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}
		cout<<"return_date_right "<<left<<" "<<right<<endl;
		left = binary_search_currency_left(left,right,set,price_currency); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}
		cout<<"return_cur_left "<<left<<" "<<right<<endl;
		right = binary_search_currency_right(left,right,set,price_currency); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}

		float candidate=0;
		if(price_minmax=="max"){
			for(int i=left;i<right+1;i++){
				if(set[i].max>candidate) candidate = set[i].max;
			}
		}
		else{
			candidate = 99999999999;
			for(int i=left;i<right+1;i++){
				if(set[i].min<candidate) candidate = set[i].min;
			} 
		}

		result<< fixed << setprecision(4)<<candidate<<'\n';
	} */
	
	// cap /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
	cout<<"sort ..."<<set.size()<<endl;  
	sort (set.begin(), set.end(), myfunction_cap); 
	string cap_date,cap_exchange; 
	file.open(argv[2], ios::in); 
	result.open("myfunction_cap.txt",ios::out);
	for(int i=0;i<set.size();i++){
		result<<set[i].date<<"\t"<<set[i].currency<<"\t"<<set[i].exchange<<"\t"<<set[i].max<<"\t"<<set[i].min<<"\t"<<set[i].cap<<'\n';
	}     
	result.close(); 
	result.open("result_3.txt",ios::out); 
	while(getline(file,str,'\n')){ 
		if(str=="end") break; 
		size_t pos = 0; 
		string delimiter ="\t"; 
		string token; 
		int left=0;
		int right=set.size()-1; 
 
		pos = str.find(delimiter);
		token = str.substr(0, pos); 
		str.erase(0, pos + delimiter.length());

		pos = str.find(delimiter);
		token = str.substr(0, pos); 
		str.erase(0, pos + delimiter.length());
		cap_date=token;
 
		pos = str.find(delimiter);
		token = str.substr(0, pos);    
		str.erase(0, pos + delimiter.length());
		cap_exchange=token; 

		cout<<cap_date<<" "<<cap_exchange<<" "<<endl<<endl;
		
		left = binary_search_date_left(left,right,set,cap_date); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}
		cout<<"return_date_left "<<left<<" "<<right<<endl;
		right = binary_search_date_right(left,right,set,cap_date); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}
		cout<<"return_date_right "<<left<<" "<<right<<endl;
		left = binary_search_exchange_left(left,right,set,cap_exchange); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}
		cout<<"return_cur_left "<<left<<" "<<right<<endl;
		right = binary_search_exchange_right(left,right,set,cap_exchange); 
		if(left==-1 || right ==-1){
			result<<"none"<<endl;
			continue;
		}

		long long int total = 0 ; 
		for(int i=left;i<right+1;i++) {
			total+=set[i].cap;
		}

		result<<total<<'\n';
	} 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	return 0;
}