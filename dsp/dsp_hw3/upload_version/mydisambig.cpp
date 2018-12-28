#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <fstream>
#include "Ngram.h"
#include "VocabMap.h"
using namespace std;
 
#define max_word_in_a_sentence 100 
#define STATE_NUM 1100
#define NEVER -50 
 
std::vector<const char*> answer;
const VocabIndex empt_context[] = {Vocab_None};

void read_in( VocabMap& map, Ngram& lm , File& test_data);
void Viterbi( VocabMap& map, Ngram& lm , VocabString* words, unsigned count);
VocabIndex getIndex(VocabString w);
 
static Vocab voc; // Vocabulary in Language Model
static Vocab vocZ, vocB; // Vocabulary ZhuYin and Big5 in ZhuYin-to-Big5 map

void read_in(VocabMap& map , Ngram& lm  , File& test_data)
{
	//printf("read in...\n"); 
	char* SAMPLE = NULL;
	int count_ = 0; 
	while(SAMPLE = test_data.getline()){
	//for(int i=0;i<2;i++){ 
		//printf("while read in...\n");
		//if(count_>0) break;
		count_++;
		VocabString content[maxWordsPerLine];
		content[0] = "<s>"; 
		unsigned count = Vocab::parseWords(SAMPLE, &content[1], maxWordsPerLine); 
		content[count+1] = "</s>"; 
		//cout<<"count = "<<count<<endl;
		Viterbi( map , lm , content, count + 2 );
	}
}


void Viterbi( VocabMap& map , Ngram& lm , VocabString* sequence, unsigned count)
{
	 
	/* Viterbi  initial */
	//printf("initializing...\n");
	LogP prob[STATE_NUM][count] = {0.0};
	vector<VocabIndex> record[STATE_NUM][count] ;
	int size[count] = {0};
 	/*
 	VocabIndex w; 
 	for(int i=0;i<STATE_NUM;i++){
 		w = vocZ.getIndex(sequence[0]);
 		record[i][1].push_back(w);
 	}
 	*/
 	//printf("initializing first line...\n");
 	Prob p; 
 	VocabIndex w; 
	VocabMapIter iter( map , vocZ.getIndex(sequence[0]) );
	for( int i = 0; iter.next(w, p) ; size[0]++ , i++ ){
		VocabIndex index = getIndex(vocB.getWord(w));
		LogP get_prob = lm.wordProb(index, empt_context);
		if(get_prob == LogP_Zero) get_prob = NEVER;
		prob[i][0] = get_prob;
		record[i][0].push_back(w);
	}
	//printf("iteratively solving...\n");
	/* Viterbi Algorithm iteratively solve  */
	//for(int i=0;i<count;i++)cout<<"size "<<i<<":"<<size[i]<<endl;
	//for(int i=0;i<size[0];i++) cout<<record[i][0][0]<<endl;
	for( int j = 1 ; j < count ; j++ ){
		Prob p; 
		VocabIndex w; 
		LogP total , tmp = 0 ;
		VocabMapIter iter( map , vocZ.getIndex(sequence[j]) );
		for( int i = 0; iter.next(w, p) ; size[j]++ , i++ ){
			VocabIndex index = getIndex(vocB.getWord(w));
			tmp = -1.0/0.0 ;
			for( int t = 0 ; t < size[j-1] ; t++ ){
				VocabIndex context[] = { getIndex(vocB.getWord(record[t][j-1][j-1])) , Vocab_None };	
				LogP get_prob = lm.wordProb(index, context);
				LogP get_prob_uni = lm.wordProb(index, empt_context);
				if(get_prob == LogP_Zero) get_prob = NEVER;
				if(get_prob_uni == LogP_Zero) get_prob_uni = NEVER;
				total = get_prob + get_prob_uni + prob[t][j-1] ;
				if( total > tmp ) {
					tmp = total ;
					record[i][j] =  record[t][j-1];
					record[i][j].push_back(w);
					prob[i][j] = total;
				}
			}
		}
	} /*
	cout<<"size : "<<endl;
	for(int i=0;i<count;i++){
		cout<<"size"<<i<<": "<<size[i]<<endl;
	}
	cout<<endl;

	cout<<"record (size) :"<<endl;
	for(int i=0;i<STATE_NUM;i++){
		for(int j=0;j<count;j++){
			cout<< record[i][j].size()<<" ";
		}
		cout<<endl;
	}
	cout<<endl;*/
	 
	/* maximize probability */
	 

	/* BackTrack from end */ 

	for( int t = 0 ; t < record[0][count-1].size() ; t++ )
		answer.push_back(vocB.getWord(record[0][count-1][t]));
}

VocabIndex getIndex(VocabString w)
{
	VocabIndex wid = voc.getIndex(w);
	if(wid == Vocab_None) return voc.getIndex(Vocab_Unknown);
	else return wid; 
}
 

int main(int argc, char *argv[])
{
	Ngram lm(voc, 2);	
	VocabMap map(vocZ, vocB); 

	//printf("1!!...\n");
	File lmFile(argv[1], "r" );
	lm.read(lmFile);
	lmFile.close();

	//printf("2!!...\n");
	File mapfile(argv[2], "r");
	map.read(mapfile);
	mapfile.close();

	//printf("3!!...\n");
	File test_data(argv[3], "r");

	//printf("4!!...\n");  
	read_in(map, lm, test_data);

	//printf("5!!...\n");
	test_data.close();

	//printf("6!!...\n");
	fstream file; 

    file.open(argv[4], ios::out);      //開啟檔案
    if(!file){
		cerr << "Can't open file!\n";
		exit(1);     //在不正常情形下，中斷程式的執行
    }
    char* a = "</s>";
    int count = 0; 
    //printf("7!!...\n");
    for(int i=0;i<answer.size();i++){
    	count++;
    	file<<answer[i];
    	if(count!=0) file<<' ';
    	if(*answer[i]==*a && count>1) {
     		file<<"\n"; 
     		count = 0;
    	}
	}

	return 0;
}