#include <stdio.h>
#include <math.h>
#include "matrix.h"

const float eps = 1e-7;

// Linear Regression loss functions : --------------------

typedef enum{
    LOSS_MSE,
    LOSS_MAE,
    LOSS_HuberLoss,
    LOSS_BCE,
    LOSS_CCE,
    LOSS_SCCE
}Loss_Type;

typedef float (*Loss_function)(const Matrix *RealOutput, const Matrix *predictedOutput, float *Parameters);
typedef void (*Gradient_Loss_function)(const Matrix *RealOutput, const Matrix *predictedOutput, Matrix *result, float *Parameters);

typedef struct
{
    Loss_Type type;

    Loss_function f;
    Gradient_Loss_function d_f;

    int n_Parameters;
    float *Parameters;
}Loss;

Loss_Type Activation_TypeOf(Loss *a){
    return a->type;
}

Loss New_Loss(Loss_Type type, float *Parameters, int n_Parameters){
    Loss result;
    result.type = type;
    if(type == LOSS_MSE){
        result.f = Mean_Squared_Error;
        result.d_f = d_Mean_Squared_Error;
        result.n_Parameters = 0;
        result.Parameters = NULL;

        return result;
    }
}

float Mean_Squared_Error(const Matrix *RealOutput, const Matrix *predictedOutput, float *Parameters){
    if(Is_Null_Matrix(RealOutput) || Is_Null_Matrix(predictedOutput))return 0.0;
    if(!matrix_same_dimensions(RealOutput, predictedOutput))return 0.0;
    if(!matrix_is_vector(RealOutput))return 0.0;

    (void)Parameters;

    int size = RealOutput->n_rows;
    float result = 0;

    for(size_t i = 0; i < size; i++){
        float error = *At(RealOutput, i , 0)-*At(predictedOutput, i ,0);
        result += error * error;
    }

    return result/size;
}
void d_Mean_Squared_Error(const Matrix *RealOutput, const Matrix *predictedOutput, Matrix *result, float *Parameters){
    if(Is_Null_Matrix(RealOutput) || Is_Null_Matrix(predictedOutput) || Is_Null_Matrix(result))return;
    if(!matrix_same_dimensions(RealOutput, predictedOutput) || !matrix_same_dimensions(RealOutput, result))return;
    if(!matrix_is_vector(RealOutput))return;

    (void)Parameters;

    int size = RealOutput->n_rows;

    for(size_t i = 0; i < size; i++){
        float error = *At(predictedOutput, i , 0)-*At(RealOutput, i ,0);
        *At(result, i, 0) = 2*error/size;
    }

    return;
}

float Mean_Absoult_Error(const Matrix *RealOutput, const Matrix *predictedOutput, float *Parameters){
    if(Is_Null_Matrix(RealOutput) || Is_Null_Matrix(predictedOutput))return 0.0;
    if(!matrix_same_dimensions(RealOutput, predictedOutput))return 0.0;
    if(!matrix_is_vector(RealOutput))return 0.0;

    (void)Parameters;

    int size = RealOutput->n_rows;
    float result = 0;

    for(size_t i = 0; i < size; i++){
        result += fabs(*At(RealOutput, i , 0)-*At(predictedOutput, i ,0));
    }

    return result/size;
}
void d_Mean_Absoult_Error(const Matrix *RealOutput, const Matrix *predictedOutput, Matrix *result, float *Parameters){
    if(Is_Null_Matrix(RealOutput) || Is_Null_Matrix(predictedOutput) || Is_Null_Matrix(result))return;
    if(!matrix_same_dimensions(RealOutput, predictedOutput) || !matrix_same_dimensions(RealOutput, result))return;
    if(!matrix_is_vector(RealOutput))return;

    (void)Parameters;

    int size = RealOutput->n_rows;

    for(size_t i = 0; i < size; i++){
        float error = *At(predictedOutput, i , 0)-*At(RealOutput, i ,0);
        *At(result, i, 0) = error >= 0 ? 1.0/(float)size :-1.0/(float)size;
    }

    return;
}

float Huber_Loss(const Matrix *RealOutput, const Matrix *predictedOutput, float *Parameters){
    if(Is_Null_Matrix(RealOutput) || Is_Null_Matrix(predictedOutput))return 0.0;
    if(!matrix_same_dimensions(RealOutput, predictedOutput))return 0.0;
    if(!matrix_is_vector(RealOutput))return 0.0;
    
    float delta = Parameters[0];

    int size = RealOutput->n_rows;
    float result = 0;

    for(size_t i = 0; i < size; i++){
        float error = fabs(*At(predictedOutput, i , 0)-*At(RealOutput, i ,0));
        result += (error <= delta) ? (0.5 * error * error) : (delta * (error - 0.5 * delta));
    }

    return result/size;
}
void d_Huber_Loss(const Matrix *RealOutput, const Matrix *predictedOutput, Matrix *result, float *Parameters){
    if(Is_Null_Matrix(RealOutput) || Is_Null_Matrix(predictedOutput) || Is_Null_Matrix(result))return;
    if(!matrix_same_dimensions(RealOutput, predictedOutput) || !matrix_same_dimensions(RealOutput, result))return;
    if(!matrix_is_vector(RealOutput))return;

    float delta = Parameters[0];

    int size = RealOutput->n_rows;

    for(size_t i = 0; i < size; i++){
        float error = *At(predictedOutput, i , 0)-*At(RealOutput, i ,0);
        float abs_error = fabs(error);
        *At(result, i, 0) = (abs_error <= delta) ? error/size : ((error > 0) ? delta : -delta) / size;
    }

    return;
}

// classification loss functions : --------------------

float Binary_Cross_Entropy_Loss(const Matrix *RealOutput, const Matrix *predictedOutput, float *Parameters){
    if(Is_Null_Matrix(RealOutput) || Is_Null_Matrix(predictedOutput))return 0.0;
    if(!matrix_same_dimensions(RealOutput, predictedOutput))return 0.0;
    if(!matrix_is_vector(RealOutput))return 0.0;

    (void)Parameters;

    int size = RealOutput->n_rows;
    float result = 0;

    for(size_t i = 0; i < size; i++){
        float p = *At(predictedOutput, i, 0);
        if(p < 0 || p > 1){
            return 0.0;
        }
        if (p < eps){
            p = eps;
        }
        if (p > 1.0 - eps){
            p = 1.0 - eps;
        }
        float error = *At(RealOutput, i , 0) * log(p) + (1 - *At(RealOutput, i , 0)) * log(1 - p);
        result += error;
    }

    return -result/size;
}
void d_Binary_Cross_Entropy_Loss(const Matrix *RealOutput, const Matrix *predictedOutput, Matrix *result, float *Parameters){
    if(Is_Null_Matrix(RealOutput) || Is_Null_Matrix(predictedOutput) || Is_Null_Matrix(result))return;
    if(!matrix_same_dimensions(RealOutput, predictedOutput) || !matrix_same_dimensions(RealOutput, result))return;
    if(!matrix_is_vector(RealOutput))return;

    (void)Parameters;

    int size = RealOutput->n_rows;

    for(size_t i = 0; i < size; i++){
        float p = *At(predictedOutput, i, 0);
        if(p < 0 || p > 1){
            return 0.0;
        }
        if (p < eps){
            p = eps;
        }
        if (p > 1.0 - eps){
            p = 1.0 - eps;
        }
        float error = -(*At(RealOutput, i , 0) / p) + (1 - *At(RealOutput, i , 0)) / (1-p);
        *At(result, i, 0) = error/size;
    }

    return;
}

float Categorical_Cross_Entropy_Loss(const Matrix *RealOutput, const Matrix *predictedOutput, float *Parameters){
    if(RealOutput->n_rows != predictedOutput->n_rows){
        printf("sizes do not match\n");
        return 0.0;
    }
    int size = RealOutput->n_rows;
    float result = 0;
    for(size_t i = 0; i < size; i++){
        for(size_t j = 0; j < RealOutput->n_cols; j++){
            if(*At(RealOutput, i, j) == 1){
                float p = *At(predictedOutput, i, j);
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

                float error = -log(p);
                result += error;
                break;
            }
        }
    }
    return result/size;
}
void d_Categorical_Cross_Entropy_Loss(const Matrix *RealOutput, const Matrix *predictedOutput, Matrix *result, float *Parameters){
    if(RealOutput->n_rows != predictedOutput->n_rows){
        return;
    }
    int size = RealOutput->n_rows;
    float result = 0;
    for(size_t i = 0; i < size; i++){
        for(size_t j = 0; j < RealOutput->n_cols; j++){
            if(*At(RealOutput, i, j) == 1){
                float p = *At(predictedOutput, i, j);
                if(p < 0 || p > 1){
                    return;
                }
                if (p < eps){
                    p = eps;
                }
                if (p > 1.0 - eps){
                    p = 1.0 - eps;
                }

                *At(result, i , j) = -1.0/(size*p);
            }
        }
    }
    return;
}

float Sparse_Categorical_Cross_Entropy_Loss(const Matrix *RealOutput, const Matrix *predictedOutput, float *Parameters){
    if(RealOutput->n_cols != 1){
        printf("real output is not a vector\n");
        return 0.0;
    }
    if(RealOutput->n_rows != predictedOutput->n_rows){
        printf("sizes do not match\n");
        return 0.0;
    }
    int size = RealOutput->n_rows;
    float result = 0;

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

        float p = *At(predictedOutput, i, index);
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

        float error = -log(p);
        result += error;
    }

    return result/size;
}
void d_Sparse_Categorical_Cross_Entropy_Loss(const Matrix *RealOutput, const Matrix *predictedOutput, Matrix *result, float *Parameters){
    if(RealOutput->n_cols != 1){
        return;
    }
    if(RealOutput->n_rows != predictedOutput->n_rows){
        return;
    }
    int size = RealOutput->n_rows;
    float result = 0;

    for(size_t i = 0; i < size; i++){
        int index = (int)*At(RealOutput, i, 0);
        if(index != *At(RealOutput, i, 0)){
            return;
        }
        if(index >= predictedOutput->n_cols){
            return;
        }

        float p = *At(predictedOutput, i, index);
        if(p < 0 || p > 1){
            return;
        }
        if (p < eps){
            p = eps;
        }
        if (p > 1.0 - eps){
            p = 1.0 - eps;
        }

        float error = -log(p);
        *At(result, i , index) = -1.0/(size*p);
    }

    return;
}