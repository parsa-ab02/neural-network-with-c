#include <stdio.h>
#include <math.h>
#include "matrix.h"
#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef enum{
    ACTIVATION_ReLU,
    ACTIVATION_LeakyReLU,
    ACTIVATION_Sigmoid,
    ACTIVATION_Tanh,
    ACTIVATION_softmax
}Type;

typedef void (*Activation_function)(const Matrix *mat, Matrix *result, float *Parameters);

typedef struct{
    Type type;

    Activation_function f;
    Activation_function d_f;

    int n_parameters;
    float *Parameters;
}Activation;

Type TypeOf(Activation *a){
    return a->type;
}

Activation New_Activation(Type type, float *Parameters, int n_parameters){
    Activation result;
    result.type = type;
    if(type == ACTIVATION_ReLU){
        result.f = ReLU;
        result.d_f = d_ReLU;
        result.n_parameters = 0;
        result.Parameters = NULL;

        return result;
    }
    if(type == ACTIVATION_LeakyReLU){
        result.f = Leaky_ReLU;
        result.d_f = d_Leaky_ReLU;
        result.n_parameters = 0;
        result.Parameters = Parameters;

        return result;
    }
}

void ReLU(const Matrix *mat, Matrix *result, float *Parameters){
    (void)Parameters;
    for(size_t i = 0; i < mat->n_rows; i++){
        for(size_t j = 0; j < mat->n_cols; j++){
            *At(result, i, j) = (*At(mat, i, j)) > 0 ? (*At(mat, i, j)) : 0;
        }
    }
}
void d_ReLU(const Matrix *mat, Matrix *result, float *Parameters){
    (void)Parameters;
    for(size_t i = 0; i < mat->n_rows; i++){
        for(size_t j = 0; j < mat->n_cols; j++){
            *At(result, i, j) = (*At(mat, i, j)) > 0 ? 1 : 0;
        }
    }
}

void Leaky_ReLU(const Matrix *mat, Matrix *result, float *Parameters){
    float Alpha = Parameters[0];
    for(size_t i = 0; i < mat->n_rows; i++){
        for(size_t j = 0; j < mat->n_cols; j++){
            *At(result, i, j) = (*At(mat, i, j)) > 0 ? (*At(mat, i, j)) : (*At(mat, i, j))*Alpha;
        }
    }
}
void d_Leaky_ReLU(const Matrix *mat, Matrix *result, float *Parameters){
    float Alpha = Parameters[0];
    for(size_t i = 0; i < mat->n_rows; i++){
        for(size_t j = 0; j < mat->n_cols; j++){
            *At(result, i, j) = (*At(mat, i, j)) > 0 ? 1 : Alpha;
        }
    }
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