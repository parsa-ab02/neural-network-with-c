#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "Activation.h"
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

typedef struct{
    int n_layers;
    int *Components;

    Matrix *Layers_Weight;
    Matrix *Layers_Bias;

    Matrix *Pre_Activations;
    Matrix *Activations;
    Matrix *Delta_Errors;
}Neural_Network;

void Set_Random_Weights(Matrix *mat){
    for(size_t i = 0; i < mat->n_rows; i++){
        for(size_t j = 0; j < mat->n_cols; j++){
            *At(mat, i, j) = rand() % 10;
        }
    }
}

Neural_Network New_Neural_Nework(int n_layers, int *Components){
    Neural_Network network;

    network.n_layers = n_layers;
    network.Components = Components;
    
    network.Layers_Weight = malloc(n_layers * sizeof(Matrix));
    network.Layers_Bias = malloc(n_layers * sizeof(Matrix));

    network.Pre_Activations = malloc(n_layers * sizeof(Matrix));
    network.Activations = malloc(n_layers * sizeof(Matrix));
    network.Delta_Errors = malloc(n_layers * sizeof(Matrix));

    for(size_t i = 1; i < network.n_layers; i++){
        network.Layers_Weight[i] = New_Matrix(network.Components[i], network.Components[i-1]);
        network.Layers_Bias[i] = New_Matrix(network.Components[i], 1);

        network.Pre_Activations[i] = New_Matrix(network.Components[i], 1);
        network.Activations[i] = New_Matrix(network.Components[i], 1);
        // if(network.activation != softmax){
        //     network.Delta_Errors[i] = New_Matrix(network.Components[i], 1);
        // }else{
        //     network.Delta_Errors[i] = New_Matrix(network.Components[i], network.Components[i]);
        // }

        Set_Random_Weights(&network.Layers_Weight[i]);
        Set_Random_Weights(&network.Layers_Bias[i]);
    }
    return network;
}

void Destroy_network(Neural_Network *network){
    for(size_t i = 1; i < network->n_layers; i++){
        destroy(&network->Layers_Weight[i]);
        destroy(&network->Layers_Bias[i]);

        destroy(&network->Pre_Activations[i]);
        destroy(&network->Activations[i]);
        destroy(&network->Delta_Errors[i]);
    }

    free(network->Layers_Weight);
    free(network->Layers_Bias);

    free(network->Pre_Activations);
    free(network->Activations);
    free(network->Delta_Errors);

    network->Layers_Weight = NULL;
    network->Layers_Bias = NULL;
    network->n_layers = 0;

    free(network->Components);
}

int Largest_Component(const Neural_Network *network){
    int Max = network->Components[1];
    for(size_t i = 0; i < network->n_layers; i++){
        if(network->Components[i] > Max) Max = network->Components[i];
    }
    return Max;
}

void Linear_Transformation(const Matrix *matrix, Matrix *Output, const Matrix *Weights, const Matrix *Bias){
    Multiplication(Weights, matrix, Output);
    Addition(Output, Bias, Output);
    return;
}

void Forward_Propagation(const Neural_Network *network,Matrix *matrix){
    network->Pre_Activations[0] = *matrix;

    for(size_t i = 1; i < network->n_layers; i++){
        Linear_Transformation(&network->Pre_Activations[i-1], &network->Pre_Activations[i], &network->Layers_Weight[i], &network->Layers_Bias[i]);

        // if(network->Activation_functions[i].f != softmax){
        //     Matrix_Activation(&network->Pre_Activations[i], &network->Activations[i], network->Activation_functions->f);
        // }else{
        //     softmax(&network->Pre_Activations[i], &network->Activations[i]);
        // }
    }
    return;
}

void Back_Propagation(const Neural_Network *network){
}

#endif