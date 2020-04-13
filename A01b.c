#include <stdio.h>
#include "A01b.h"

void random_index(matrix *TARGET, size_t limit, size_t set) {
	TARGET->sizeX   = limit / set;
	TARGET->sizeY   = set;
	TARGET->element = realloc(TARGET->element, sizeof(double) * limit);

	struct timeval tme;
	gettimeofday(&tme, 0);
	srand((unsigned) tme.tv_usec);

	double buff;
	for(int i = 0; i < limit; i++) {
		if(i == 0) TARGET->element[i] = rand() % limit;
		else {
jump:
			buff = rand() % limit;
			for(int x = 0; x < i; x++) if(buff == TARGET->element[x]) goto jump;
			TARGET->element[i] = buff;
		}
	}
}

void REwriter_matrix(matrix *TARGET, matrix META, matrix position) {
	TARGET->sizeX   = position.sizeY;
	TARGET->sizeY   = META.sizeY;
	TARGET->element = realloc(TARGET->element, sizeof(double) * position.sizeY * META.sizeY);
	double *TARGET_ptr	 = TARGET->element,
		   *META_ptr	 = META.element,
		   *position_ptr = position.element;

	for(int is = 0; is < position.sizeY; is++) {
		META_ptr += (int)*position_ptr * META.sizeY;
		for(int i = 0; i < META.sizeY; i++) *TARGET_ptr++ = *META_ptr++;

		META_ptr -= (int)*position_ptr++ * META.sizeY + META.sizeY;
	}
}

int main() {
	matrix INPUT,  TRAIN,  TRAIN_buff;
	init (&INPUT, &TRAIN, &TRAIN_buff, (matrix *) 0);

	read_matrix(&INPUT, "./INFILE");
	read_matrix(&TRAIN, "./TRFILE");

	int NODE_size[] = { INPUT.sizeY, 100, 50, 25, TRAIN.sizeY },
		NODE_length = sizeof(NODE_size) / sizeof(int),
		BATCH_size  = 9,
		reseter = 0, training_done = 1, i;

	double training_target = 1,
		   *last_layer, error_total = 0, error_average;

	matrix shuffled_index, layer_sigmoid,
		   LAYER [NODE_length],
		   WEIGHT[NODE_length - 1],
		   delta [NODE_length - 1];
		   init (&shuffled_index, &layer_sigmoid, (matrix *) 0);
		   inits(LAYER,  NODE_length);
		   inits(WEIGHT, NODE_length - 1);
		   inits(delta,  NODE_length - 1);

		   random_index(&shuffled_index, INPUT.sizeX, BATCH_size);

		   for(i = 0; i < NODE_length - 1; i++) 
			   MOD_random(&WEIGHT[i], NODE_size[i], NODE_size[i + 1]);

		   for(;; training_done++) {
			   REwriter_matrix(&LAYER[0],   INPUT, shuffled_index);
			   REwriter_matrix(&TRAIN_buff, TRAIN, shuffled_index);
			   shuffled_index.element += BATCH_size;
			   reseter += BATCH_size;

			   for(i = 0; i < NODE_length - 1; i++) {
				   mul_matrix    (&LAYER[i + 1], LAYER[i], WEIGHT[i]);
				   sigmoid_matrix(&LAYER[i + 1], LAYER[i + 1]);
			   }

			   for(i = NODE_length - 2; i >= 0; i--)
				   if(i == NODE_length - 2) {
					   sub_matrix	 (&delta[i], TRAIN_buff, LAYER[i + 1]);
					   sigmoid_dx_matrix(&layer_sigmoid,	 LAYER[i + 1]);
					   mul_matrix_alt   (&delta[i], delta[i], layer_sigmoid);
				   } else {
					   MOD_dot_matrix   (&delta[i], delta[i + 1], WEIGHT[i + 1]);
					   sigmoid_dx_matrix(&layer_sigmoid, LAYER[i + 1]);
					   mul_matrix_alt   (&delta[i], delta[i], layer_sigmoid);
				   }

			   for(i = 0; i < NODE_length - 1; i++) { 
				   mirror_matrix (&delta [i], delta[i]);
				   mirror_matrix (&LAYER [i], LAYER[i]);
				   MOD_dot_matrix(&delta [i], LAYER[i], delta [i]);
				   add_matrix    (&WEIGHT[i], delta[i], WEIGHT[i]);
			   }

			   last_layer = LAYER[NODE_length - 1].element;
			   for(i = 0; i < BATCH_size * NODE_size[NODE_length - 1]; i++) 
				   error_total += 0.5 * pow(TRAIN_buff.element[i] - last_layer[i], 2);
			   error_average = error_total / training_done;
			   printf("\r\x1B[38;5;12mloss %.6f", error_average);

			   if(reseter == INPUT.sizeX) {
				   shuffled_index.element -= reseter;
				   reseter = 0;
				   if(error_average < training_target) break;	
			   }
		   }
		   printf("\n");
		   goto done;
		   read_matrix(&LAYER[0], "./INFILE");
		   for(i = 0; i < NODE_length - 1; i++) { 
			   mul_matrix    (&LAYER[i + 1], LAYER[i], WEIGHT[i]);
			   sigmoid_matrix(&LAYER[i + 1], LAYER[i + 1]);
		   }
		   printm(LAYER[0],		   "IN",  0.5);
		   printm(LAYER[NODE_length - 1], "OUT", 0.5);
done:
		   WEIGHT[0].number = NODE_length - 1;
		   write_matrixs(WEIGHT, "WEIGHT");

		   clear (&INPUT, &TRAIN, &TRAIN_buff, 
				   &shuffled_index, &layer_sigmoid, (matrix *) 0);
		   clears(LAYER,  NODE_length);
		   clears(WEIGHT, NODE_length - 1);
		   clears(delta,  NODE_length - 1);
}
