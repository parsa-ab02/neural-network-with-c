#include <stdio.h>
#include <math.h>
#include "matrix.h"
#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef enum{
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    softmax
}Type;

typedef struct{
    Type type;


}Activation;

Type TypeOf(Activation *a){
    return a->type;
}

float ReLU(float value){
    return value > 0 ? value : 0;
}
float d_ReLU(float value){
    return value > 0 ? 1.0 : 0;
}

float LeakyReLU(float value , float alpha){
    return value > 0 ? value : alpha*value;
}
float d_LeakyReLU(float value , float alpha){
    return value > 0 ? 1.0 : alpha;
}

float Sigmoid(float value){
    return 1/(1+exp(-value));
}
float d_Sigmoid(float value){
    float s = Sigmoid(value);
    return s*(1-s);
}

float Tanh(float value){
    return tanh(value);
}
float d_Tanh(float value){
    return 1 - pow(tanh(value), 2);
}

void Matrix_Activation(const Matrix *mat, Matrix *result, float (*f)(float)){
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
    float max = *At(mat, 0 , 0);
    for(size_t i = 0; i < mat->n_rows; i++){
        if(*At(mat, i , 0) > max){
        max = *At(mat, i , 0);
        }
    }

    float sum = 0;
    for(size_t i = 0; i < mat->n_rows; i++){
        float temp = exp(*At(mat,i,0) - max);
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
        float si = *At(softmax_matrix, i, 0);
        for(size_t j = 0; j < softmax_matrix->n_rows; j++){
            float sj = *At(softmax_matrix, j, 0);
            if(i == j){
                *At(result, i, j) = si * (1 - si);
            }else{
                *At(result, i, j) = -si * sj;
            }
        }
    }
}

#endif