#include <stdio.h>
#include "A01b.h"

int main() {
    matrix *WEIGHT = malloc(1);
    read_matrixs(&WEIGHT, "./WEIGHT");

    matrix LAYER [WEIGHT[0].number + 1];
    inits (LAYER, WEIGHT[0].number + 1);

    read_matrix(&LAYER[0], "./INFILE");
    printf("%d %d", LAYER[0].sizeX, LAYER[0].sizeY);
    for(int i = 0; i < WEIGHT[0].number; i++) { 
	mul_matrix    (&LAYER[i + 1], LAYER[i], WEIGHT[i]);
	sigmoid_matrix(&LAYER[i + 1], LAYER[i + 1]);
    }
    printm(LAYER[WEIGHT[0].number], "OUT", 0.5);

    clears(WEIGHT, WEIGHT[0].number);
    free(WEIGHT);
    return 0;
}
