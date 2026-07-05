#include <stdio.h>
#include <math.h>
#include "matrix.h"

const double eps = 1e-15;

// Linear Regression loss functions : --------------------

typedef struct
{
    
}Loss;


double Mean_Squared_Error(const Matrix *RealOutput, const Matrix *predictedOutput){
    if(RealOutput->n_cols != 1 || predictedOutput->n_cols != 1){
        printf("real output or predicted output are not a vector\n");
        return 0.0;
    }
    if(RealOutput->n_rows != predictedOutput->n_rows){
        printf("real output and predicted out put are not the same size\n");
        return 0.0;
    }
    int size = RealOutput->n_rows;
    double result = 0;

    for(size_t i = 0; i < size; i++){
        double error = *At(RealOutput, i , 0)-*At(predictedOutput, i ,0);
        result += error * error;
    }

    return result/size;
}

double Mean_Absoult_Error(const Matrix *RealOutput, const Matrix *predictedOutput){
    if(RealOutput->n_cols != 1 || predictedOutput->n_cols != 1){
        printf("real output or predicted output are not a vector\n");
        return 0.0;
    }
    if(RealOutput->n_rows != predictedOutput->n_rows){
        printf("real output and predicted out put are not the same size\n");
        return 0.0;
    }
    int size = RealOutput->n_rows;
    double result = 0;

    for(size_t i = 0; i < size; i++){
        result += fabs(*At(RealOutput, i , 0)-*At(predictedOutput, i ,0));
    }

    return result/size;
}

double Huber_Loss(const Matrix *RealOutput, const Matrix *predictedOutput, double delta){
    if(RealOutput->n_cols != 1 || predictedOutput->n_cols != 1){
        printf("real output or predicted output are not a vector\n");
        return 0.0;
    }
    if(RealOutput->n_rows != predictedOutput->n_rows){
        printf("real output and predicted out put are not the same size\n");
        return 0.0;
    }
    int size = RealOutput->n_rows;
    double result = 0;

    for(size_t i = 0; i < size; i++){
        double error = fabs(*At(RealOutput, i , 0)-*At(predictedOutput, i ,0));
        if(error <= delta){
            result += (0.5*error * error);
        }else{
            result += delta*(error - 0.5*delta);
        }
    }

    return result/size;
}

// classification loss functions : --------------------

double Binary_Cross_Entropy_Loss(const Matrix *RealOutput, const Matrix *predictedOutput){
    if(RealOutput->n_cols != 1 || predictedOutput->n_cols != 1){
        printf("real output or predicted output are not a vector\n");
        return 0.0;
    }
    if(RealOutput->n_rows != predictedOutput->n_rows){
        printf("real output and predicted out put are not the same size\n");
        return 0.0;
    }
    int size = RealOutput->n_rows;
    double result = 0;

    for(size_t i = 0; i < size; i++){
        double p = *At(predictedOutput, i, 0);
        if(p < 0 || p > 1){
            printf("probabilities are invalid! it must be  between 0 to 1\n");
            return 0.0;
        }
        if (p < eps){
            p = eps;
        }
        if (p > 1.0 - eps){
            p = 1.0 - eps;
        }
        double error = *At(RealOutput, i , 0) * log(p) + (1 - *At(RealOutput, i , 0)) * log(1 - p);
        result += error;
    }

    return -result/size;
}

double Categorical_Cross_Entropy_Loss(const Matrix *RealOutput, const Matrix *predictedOutput){
    if(RealOutput->n_rows != predictedOutput->n_rows){
        printf("sizes do not match\n");
        return 0.0;
    }
    int size = RealOutput->n_rows;
    double result = 0;
    for(size_t i = 0; i < size; i++){
        for(size_t j = 0; j < RealOutput->n_cols; j++){
            if(*At(RealOutput, i, j) == 1){
                double p = *At(predictedOutput, i, j);
                if(p < 0 || p > 1){
                    printf("probabilities are invalid! it must be  between 0 to 1\n");
                    return 0.0;
                }
                if (p < eps){
                    p = eps;
                }
                if (p > 1.0 - eps){
                    p = 1.0 - eps;
                }

                double error = -log(p);
                result += error;
                continue;
            }
        }
    }
    return result/size;
}

double Sparse_Categorical_Cross_Entropy_Loss(const Matrix *RealOutput, const Matrix *predictedOutput){
    if(RealOutput->n_cols != 1){
        printf("real output is not a vector\n");
        return 0.0;
    }
    if(RealOutput->n_rows != predictedOutput->n_rows){
        printf("sizes do not match\n");
        return 0.0;
    }
    int size = RealOutput->n_rows;
    double result = 0;

    for(size_t i = 0; i < size; i++){
        int index = (int)*At(RealOutput, i, 0);
        if(index != *At(RealOutput, i, 0)){
            printf("output vector is invalid! its type is not int\n");
            return 0.0;
        }
        if(index >= predictedOutput->n_cols){
            printf("output vector is invalid! its indexing doesn't match the size of predictions matrix\b");
            return 0.0;
        }

        double p = *At(predictedOutput, i, index);
        if(p < 0 || p > 1){
            printf("probabilities are invalid! it must be  between 0 to 1\n");
            return 0.0;
        }
        if (p < eps){
            p = eps;
        }
        if (p > 1.0 - eps){
            p = 1.0 - eps;
        }

        double error = -log(p);
        result += error;
    }

    return result/size;
}