#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdlib.h>

#include "matrix.h"
#include "Activation.h"
#include "Loss.h"

typedef struct {
    Matrix weight;
    Matrix bias;

    Matrix pre_activation;
    Matrix post_activation;
    Matrix delta_error;

    Activation activation;
} Layer;

typedef struct {
    int n_layers;
    Layer *layers;
} Neural_Network;

void Set_Random_Weights(Matrix *mat){
    if (mat == NULL || mat->Data == NULL) {
        return;
    }

    for (int i = 0; i < mat->n_rows; i++) {
        for (int j = 0; j < mat->n_cols; j++) {
            *At(mat, i, j) = (float)(rand() % 10);
        }
    }
}

static void Destroy_Layer(Layer *layer){
    if (layer == NULL) {
        return;
    }

    destroy(&layer->weight);
    destroy(&layer->bias);
    destroy(&layer->pre_activation);
    destroy(&layer->post_activation);
    destroy(&layer->delta_error);

    layer->activation.f = NULL;
    layer->activation.d_f = NULL;
    layer->activation.Parameters = NULL;
    layer->activation.n_Parameters = 0;
}

static int Initialize_Layer(
    Layer *layer,
    int input_size,
    int output_size,
    const Activation *activation
){
    if (layer == NULL || input_size <= 0 || output_size <= 0) {
        return 0;
    }

    layer->weight = New_Matrix(output_size, input_size);
    layer->bias = New_Matrix(output_size, 1);
    layer->pre_activation = New_Matrix(output_size, 1);
    layer->post_activation = New_Matrix(output_size, 1);
    layer->delta_error = New_Matrix(output_size, 1);

    if (Is_Null_Matrix(&layer->weight) ||
        Is_Null_Matrix(&layer->bias) ||
        Is_Null_Matrix(&layer->pre_activation) ||
        Is_Null_Matrix(&layer->post_activation) ||
        Is_Null_Matrix(&layer->delta_error)) {
        Destroy_Layer(layer);
        return 0;
    }

    layer->activation = activation != NULL
        ? *activation
        : New_Activation(ACTIVATION_ReLU, NULL, 0);

    if (layer->activation.f == NULL || layer->activation.d_f == NULL) {
        Destroy_Layer(layer);
        return 0;
    }

    Set_Random_Weights(&layer->weight);
    Set_Random_Weights(&layer->bias);
    return 1;
}

static int Apply_Layer_Derivative(Layer *layer){
    if (layer == NULL) {
        return 0;
    }

    Matrix derivative = New_Matrix(
        layer->pre_activation.n_rows,
        layer->pre_activation.n_cols
    );
    if (Is_Null_Matrix(&derivative)) {
        return 0;
    }

    derivative_activate(
        &layer->activation,
        &layer->pre_activation,
        &derivative
    );

    for (int i = 0; i < layer->delta_error.n_rows; i++) {
        for (int j = 0; j < layer->delta_error.n_cols; j++) {
            *At(&layer->delta_error, i, j) *= *At(&derivative, i, j);
        }
    }

    destroy(&derivative);
    return 1;
}

Neural_Network New_Neural_Network(
    int n_layers,
    const int *components,
    const Activation *activations
){
    Neural_Network network = {0, NULL};

    if (n_layers <= 0 || components == NULL) {
        return network;
    }

    network.layers = calloc((size_t)n_layers, sizeof(*network.layers));
    if (network.layers == NULL) {
        return network;
    }

    for (int i = 0; i < n_layers; i++) {
        const Activation *activation = activations != NULL
            ? &activations[i]
            : NULL;

        if (!Initialize_Layer(
                &network.layers[i],
                components[i],
                components[i + 1],
                activation)) {
            for (int j = 0; j <= i; j++) {
                Destroy_Layer(&network.layers[j]);
            }
            free(network.layers);
            network.layers = NULL;
            return network;
        }
    }

    network.n_layers = n_layers;
    return network;
}

void Destroy_Neural_Network(Neural_Network *network){
    if (network == NULL) {
        return;
    }

    for (int i = 0; i < network->n_layers; i++) {
        Destroy_Layer(&network->layers[i]);
    }

    free(network->layers);
    network->layers = NULL;
    network->n_layers = 0;
}

int Largest_Component(const Neural_Network *network){
    if (network == NULL || network->layers == NULL || network->n_layers <= 0) {
        return 0;
    }

    int largest = network->layers[0].weight.n_cols;
    for (int i = 0; i < network->n_layers; i++) {
        if (network->layers[i].weight.n_rows > largest) {
            largest = network->layers[i].weight.n_rows;
        }
    }
    return largest;
}

void Linear_Transformation(
    const Matrix *input,
    Matrix *output,
    const Matrix *weights,
    const Matrix *bias
){
    if (input == NULL || output == NULL || weights == NULL || bias == NULL) {
        return;
    }

    Multiplication(weights, input, output);
    Addition(output, bias, output);
}

int Forward_Propagation(
    Neural_Network *network,
    const Matrix *input
){
    if (network == NULL || network->layers == NULL ||
        input == NULL || network->n_layers <= 0) {
        return 0;
    }

    const Matrix *current = input;

    for (int i = 0; i < network->n_layers; i++) {
        Layer *layer = &network->layers[i];

        if (current->n_rows != layer->weight.n_cols || current->n_cols != 1) {
            return 0;
        }

        Linear_Transformation(
            current,
            &layer->pre_activation,
            &layer->weight,
            &layer->bias
        );

        Activate(
            &layer->activation,
            &layer->pre_activation,
            &layer->post_activation
        );

        current = &layer->post_activation;
    }

    return 1;
}

int Back_Propagation(
    Neural_Network *network,
    const Matrix *input,
    const Matrix *target,
    const Loss *loss,
    float learning_rate
){
    if (network == NULL || network->layers == NULL ||
        input == NULL || target == NULL || loss == NULL || loss->d_f == NULL ||
        network->n_layers <= 0 || learning_rate < 0.0f) {
        return 0;
    }

    for (int i = 0; i < network->n_layers; i++) {
        if (network->layers[i].activation.type == ACTIVATION_softmax) {
            return 0;
        }
    }

    Layer *output_layer = &network->layers[network->n_layers - 1];

    if (target->n_rows != output_layer->post_activation.n_rows ||
        target->n_cols != output_layer->post_activation.n_cols) {
        return 0;
    }

    for (int i = 0; i < output_layer->delta_error.n_rows; i++) {
        for (int j = 0; j < output_layer->delta_error.n_cols; j++) {
            *At(&output_layer->delta_error, i, j) = 0.0f;
        }
    }
    loss->d_f(
        target,
        &output_layer->post_activation,
        &output_layer->delta_error,
        loss->Parameters
    );

    if (!Apply_Layer_Derivative(output_layer)) {
        return 0;
    }

    for (int layer_index = network->n_layers - 1;
         layer_index >= 0;
         layer_index--) {
        Layer *layer = &network->layers[layer_index];
        const Matrix *previous_activation = layer_index == 0
            ? input
            : &network->layers[layer_index - 1].post_activation;

        if (layer_index > 0) {
            Layer *previous_layer = &network->layers[layer_index - 1];

            for (int col = 0; col < layer->weight.n_cols; col++) {
                float sum = 0.0f;
                for (int row = 0; row < layer->weight.n_rows; row++) {
                    sum += *At(&layer->weight, row, col) *
                           *At(&layer->delta_error, row, 0);
                }
                *At(&previous_layer->delta_error, col, 0) = sum;
            }

            if (!Apply_Layer_Derivative(previous_layer)) {
                return 0;
            }
        }

        for (int row = 0; row < layer->weight.n_rows; row++) {
            for (int col = 0; col < layer->weight.n_cols; col++) {
                *At(&layer->weight, row, col) -= learning_rate *
                    *At(&layer->delta_error, row, 0) *
                    *At(previous_activation, col, 0);
            }
        }

        for (int row = 0; row < layer->bias.n_rows; row++) {
            *At(&layer->bias, row, 0) -= learning_rate *
                *At(&layer->delta_error, row, 0);
        }
    }

    return 1;
}

#endif