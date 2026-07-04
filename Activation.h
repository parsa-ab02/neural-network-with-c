#include <stdio.h>
#include <math.h>
#include "matrix.h"
#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef struct{
    double (*f)(double);
    double (*d_f)(double);
}Activation;

double ReLU(double value){
    if(value <= 0){
        return 0;
    }return value;
}
double d_ReLU(double value){
    if(value <= 0){
        return 0;
    }return 1;
}

double Max(double value1 , double value2){
    return value1 > value2 ? value1 : value2;
}

double LeakyReLU(double value , double alpha){
    return Max(value , alpha*value);
}
double d_LeakyReLU(double value , double alpha){
    return value > 0 ? 1.0 : alpha;
}

double Sigmoid(double value){
    return 1/(1+exp(-value));
}
double d_Sigmoid(double value){
    double s = Sigmoid(value);
    return s*(1-s);
}

double Tanh(double value){
    return tanh(value);
}
double d_Tanh(double value){
    return 1 - pow(tanh(value), 2);
}

void Matrix_Activation(const Matrix *mat, Matrix *result, double (*f)(double)){
    for(size_t i = 0; i < mat->n_rows; i++){
        for(size_t j = 0; j < mat->n_cols; j++){
            *At(result, i, j) = f(*At(mat, i, j));
        }
    }
}

void softmax(const Matrix *mat, Matrix *result){
    if(result->n_rows != mat->n_rows || result->n_cols != 1){
        return;
    }
    double max = *At(mat, 0 , 0);
    for(size_t i = 0; i < mat->n_rows; i++){
        if(*At(mat, i , 0) > max){
        max = *At(mat, i , 0);
        }
    }

    double sum = 0;
    for(size_t i = 0; i < mat->n_rows; i++){
        double temp = exp(*At(mat,i,0) - max);
        *At(result, i, 0) = temp;
        sum += temp;
        
    }

    for(size_t i = 0; i < mat->n_rows; i++){
        *At(result, i, 0) /= sum;
    }
    return;
}
void d_Softmax(const Matrix *softmax_matrix, Matrix *result){
    if(result->n_rows != softmax_matrix->n_rows || result->n_cols != softmax_matrix->n_rows){
    return;
}
    for(size_t i = 0; i < softmax_matrix->n_rows; i++){
        double si = *At(softmax_matrix, i, 0);
        for(size_t j = 0; j < softmax_matrix->n_rows; j++){
            double sj = *At(softmax_matrix, j, 0);
            if(i == j){
                *At(result, i, j) = si * (1 - si);
            }else{
                *At(result, i, j) = -si * sj;
            }
        }
    }
}

Activation ReLU_Activation = {
    .f = ReLU,
    .d_f = d_ReLU
};

Activation Sigmoid_Activation = {
    .f = Sigmoid,
    .d_f = d_Sigmoid
};

Activation Tanh_Activation = {
    .f = Tanh,
    .d_f = d_Tanh
};

Activation softmax_Activation = {
    .f = softmax,
    .d_f = d_Softmax
};

#endif