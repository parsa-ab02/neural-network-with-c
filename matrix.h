#include <stdio.h>
#include <stdlib.h>
#ifndef MATRIX_H
#define MATRIX_H

typedef struct{
    int n_rows;
    int n_cols;
    int capacity;
    double *Data;
}Matrix;

Matrix Null_Matrix(){
    Matrix mat;
    mat.n_rows = 0;
    mat.n_cols = 0;
    mat.capacity = 0;
    mat.Data = NULL;
    return mat;
}

int Is_Null_Matrix(Matrix mat){
    return mat.n_rows == 0 && mat.n_cols == 0 && mat.capacity == 0 && mat.Data == NULL;
}

Matrix New_Matrix(int n_rows , int n_cols){
    Matrix mat;
    mat.n_rows = n_rows;
    mat.n_cols = n_cols;
    mat.capacity = n_rows*n_cols;
    mat.Data = malloc(mat.capacity * sizeof(double));

    if(mat.Data == NULL){
        return Null_Matrix();
    }
    return mat;
}

void destroy(Matrix *mat){
    free(mat->Data);
    *mat = Null_Matrix();
}

double *At(Matrix *mat, int i, int j){
    return &mat->Data[i*mat->n_cols+j];
}

void Multiplication(const Matrix *mat1, const Matrix *mat2, Matrix *result){
    if(mat1->n_cols != mat2->n_rows){
        printf("ERROR : Matrix sizes are not valid\n");
        return;
    }
    if(result->n_rows !=  mat1->n_rows || result->n_cols != mat2->n_cols){
        printf("ERROR : Result sizes are not valid\n");
        return;
    }

    for (size_t i = 0; i < mat1->n_rows; i++){
        for (size_t j = 0; j < mat2->n_cols; j++){
            double sum = 0;
            for (size_t k = 0; k < mat1->n_cols; k++)
            {
                sum += (*At(mat1 , i , k)) * (*At(mat2 , k , j));
            }
            *At(result , i , j) = sum;
        }
    }
    return;
}

void Addition(const Matrix *mat1, const Matrix *mat2 , Matrix *result){
    if(mat1->n_rows != mat2->n_rows || mat1->n_cols != mat2->n_cols){
        printf("ERROR : Matrixes sizes are not valid\n");
        return;
    }
    if(result->n_rows !=  mat1->n_rows || result->n_cols != mat2->n_cols){
        printf("ERROR : Result sizes are not valid\n");
        return;
    }
    for(size_t i = 0; i < mat1->n_rows; i++){
        for(size_t j = 0; j < mat1->n_cols; j++){
            *At(result , i , j) = *At(mat1, i, j) + *At(mat2, i, j);
        }
    }
    return;
}

void Transpose(const Matrix *matrix, Matrix *result){
    if(matrix->n_rows != result->n_cols || matrix->n_cols != result->n_rows){
        printf("ERROR : Result sizes are not valid\n");
        return;
    }
    for(size_t i = 0; i < matrix->n_rows; i++){
        for(size_t j = 0; j < matrix->n_cols; j++){
            *At(result, j, i) = *At(matrix, i, j);
        }
    }
    return;
}

void Element_Wise_multiplication(const Matrix *mat1, const Matrix *mat2, Matrix *result){
    if(mat1->n_rows != mat2->n_rows || mat1->n_cols != mat2->n_cols){
        printf("ERROR : Matrixes sizes are not valid\n");
        return;
    }
    if(result->n_rows !=  mat1->n_rows || result->n_cols != mat2->n_cols){
        printf("ERROR : Result sizes are not valid\n");
        return;
    }
    for(size_t i = 0; i < mat1->n_rows; i++){
        for(size_t j = 0; j < mat1->n_cols; j++){
            *At(result , i , j) = (*At(mat1, i, j)) * (*At(mat2, i, j));
        }
    }
    return;
}

void print_matrix(const Matrix *matrix){
    for(size_t i = 0; i < matrix->n_rows; i++){
        for(size_t j = 0; j < matrix->n_cols; j++){
            printf("%d ",*At(matrix, i, j));
        }
        printf("\n");
    }
}

#endif