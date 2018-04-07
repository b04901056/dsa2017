#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <new>
#include <iostream>
#include "matrix.h"
using namespace std;

Matrix::Matrix(const int& r, const int& c) //constructor
{
	row = r;
	col = c;
	array = new double* [r];
	for(int i=0;i<row;i++){
		array[i] = new double [col];
	}
}

Matrix::Matrix(const Matrix& rhs) //copy constructor
{
	row = rhs.row; 
	col = rhs.col;
	array = new double* [row];
	for(int i=0;i<row;i++){
		array[i] = new double [col];
	}
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			array[i][j] = rhs[i][j];
		}
	}
}

Matrix::~Matrix() //destructor
{
	for(int i=0;i<row;i++){
		delete [] array[i];
	}
	delete array;
}

double* & Matrix::operator [](const int& idx) const
{
	return array[idx];
}

Matrix Matrix::operator =(const Matrix& rhs) // assignment constructor
{
	if(this!=&rhs){
		for(int i=0;i<row;i++){
			delete [] array[i];
		}
		delete array;
		row = rhs.row; 
		col = rhs.col;
		array = new double* [row];
		for(int i=0;i<row;i++){
			array[i] = new double [col];
		}
		for(int i=0;i<row;i++){
			for(int j=0;j<col;j++){
				array[i][j] = rhs[i][j];
			}
		}	
	}
	return *this;
}

Matrix Matrix::operator -() const
{ 
	Matrix tmp(row,col);
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			tmp[i][j] = array[i][j] * (-1);
		}
	}
	return tmp;
}

Matrix Matrix::operator +() const
{ 
	Matrix tmp(row,col);
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			tmp[i][j] = array[i][j] ;
		}
	}
	return tmp;
}

Matrix Matrix::operator -(const Matrix& rhs) const
{
	Matrix tmp(rhs.row,rhs.col);
	for(int i=0;i<tmp.row;i++){
		for(int j=0;j<tmp.col;j++){
			tmp[i][j] = this->array[i][j] - rhs[i][j];
		}
	}
	return tmp;
}

Matrix Matrix::operator +(const Matrix& rhs) const
{
	Matrix tmp(rhs.row,rhs.col);
	for(int i=0;i<tmp.row;i++){
		for(int j=0;j<tmp.col;j++){
			tmp[i][j] = this->array[i][j] + rhs[i][j];
		}
	}
	return tmp;
}

Matrix Matrix::operator *(const Matrix& rhs) const
{ 
	Matrix tmp(row,rhs.col); 
	Matrix first =(*this) ;
	for(int i=0;i<tmp.row;i++){
		for(int j=0;j<tmp.col;j++){
			double sum = 0;
			for(int k=0;k<col;k++){
				sum += first[i][k] * rhs[k][j] ;
			}
			tmp[i][j] = sum ; 
		}
	}
	return tmp;
}

Matrix Matrix::operator /(const Matrix& rhs) const
{
	Matrix inv = rhs.inverse(); 
	Matrix tmp = (*this) * inv ;
	return tmp;
}

void dev(double **a , int num , double dev ,int col){
    for(int i=0;i<col*2;i++){
        a[num][i]/=dev; 
    }
}
void change(double **a , int num1 , int num2 , int col ){
    for(int i=0;i<col*2;i++){
        double tmp = a[num1][i];
        a[num1][i] = a[num2][i]; 
        a[num2][i] = tmp; 
    }
}
Matrix Matrix::inverse() const
{ 
    double** a;
    a = new double* [col];
    for(int i=0;i<col;i++){
        a[i]=new double[col*2];
    } 
    for(int i=0;i<col;i++){
        for(int j=0;j<col;j++){
            a[i][j] = array[i][j];
        }
    }
    for(int i=0;i<col;i++){
        for(int j=col;j<col*2;j++){
            if(i==j-col) a[i][j]=1;
            else a[i][j]=0;
        }
    }  
    for(int i=0;i<col;i++){

    }
    ///start transformation
    for(int i=0;i<col;i++){
    	if(a[i][i]==0){
            for(int j=i+1;j<col;j++){
                if(a[j][i]!=0){
                    change(a,i,j,col);
                    break;
                }
            }
        }
        dev(a,i,a[i][i],col);
        for(int j=0;j<col;j++){
            if(i==j)continue;
            double dev = a[j][i]/a[i][i];
            for(int k=i;k<col*2;k++){
                a[j][k]-=a[i][k]*dev;
            }
        }
    }  
    Matrix tmp(col,col);
    for(int i=0;i<col;i++){
        for(int j=0;j<col;j++){
            tmp[i][j] = a[i][j+col];
    	} 
    } 
    for(int i=0;i<col;i++){
    	delete [] a[i];
    }
    delete a;
    
    return tmp;
}

void Matrix::read(const char* fn)
{
	int r, c;
	FILE *fp = fopen(fn, "r");
	if(fp == NULL){
		printf("read file [%s] error\n", fn);
		exit(0);
	}
	fscanf(fp, "%d%d", &r, &c);
	Matrix tmp(r, c);
	for(int i = 0 ; i < r ; i++)
		for(int j = 0 ; j < c ; j++)
			fscanf(fp, "%lf", &tmp.array[i][j]);
	fclose(fp);
	*this = tmp;
}

void Matrix::write(const char* fn)
{
	FILE *fp = fopen(fn, "w");
	if(fp == NULL){
		printf("write file [%s] error\n", fn);
		exit(0);
	}
	fprintf(fp, "%d %d\n", row, col);
	for(int i = 0 ; i < row ; i++)
		for(int j = 0 ; j < col ; j++)
			fprintf(fp, "%lf%c", array[i][j], " \n"[j==col-1]);
	fclose(fp);
}

void Matrix::print() const
{
	for(int i = 0 ; i < row ; i++)
		for(int j = 0 ; j < col ; j++)
			printf("%lf%c", array[i][j], " \n"[j==col-1]);
}