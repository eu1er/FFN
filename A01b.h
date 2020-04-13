#ifndef _A01b_H
#define _A01b_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

typedef struct{
    size_t sizeX,
	   sizeY,
	   number;
    double *element;
} matrix;

void init_(matrix *TARGET) {
    TARGET->sizeX   = 0;
    TARGET->sizeY   = 0;
    TARGET->element = malloc(1);
}

void clear_(matrix *TARGET)              { free(TARGET->element); }
void inits (matrix *TARGET, size_t size) { for(int i = 0; i < size; i++) init_ (&TARGET[i]); }
void clears(matrix *TARGET, size_t size) { for(int i = 0; i < size; i++) clear_(&TARGET[i]); }

void init(matrix *TARGET, ...) {
    va_list list;
    va_start(list, TARGET);

    while(TARGET != 0)  {
	init_((matrix *) TARGET);
	TARGET = va_arg(list, matrix *);
    }

    va_end(list);
}

void clear(matrix *TARGET, ...) {
    va_list list;
    va_start(list, TARGET);

    while(TARGET != 0)  {
	clear_((matrix *) TARGET);
	TARGET = va_arg(list, matrix *);
    }

    va_end(list);
}

void printm(matrix INPUT, char *text, double threshold) {
    printf("\n\x1B[38;5;7m%s %dx%d\n", text, INPUT.sizeX, INPUT.sizeY);
    double buff;
    for(int i = 0; i < INPUT.sizeX * INPUT.sizeY; i++) {
	(INPUT.element[i] >= threshold)? printf("\x1B[38;5;2m"): printf("\x1B[38;5;1m");

	buff = INPUT.element[i];
	if(buff >= 0) printf(" %.2f ", buff);
	else printf("%.2f ", buff);

	if ( (i + 1) % INPUT.sizeY == 0 ) printf("\n");
    }
}

void MOD_random(matrix *TARGET, size_t sizeX, size_t sizeY) {
    TARGET->sizeX   = sizeX;
    TARGET->sizeY   = sizeY;
    TARGET->element = realloc(TARGET->element, sizeof(double) * sizeX *sizeY);

    struct timeval tme;
    gettimeofday(&tme, 0);
    srand((unsigned) tme.tv_usec);
    
    double buff;
    for(int i = 0; i < sizeX * sizeY; i++) {
	buff = (double)rand() / (double)(RAND_MAX);
	TARGET->element[i] = rand() % 2 == 0? buff: -buff;
    }
}

#define sigmoid(x) 1 / (1 + exp(-x))
#define sigmoid_dx(x) x * (1 - x)

void sigmoid_matrix(matrix *TARGET, matrix INPUT) {
    size_t size = INPUT.sizeX * INPUT.sizeY, i = 0;

    TARGET->sizeX   = INPUT.sizeX;
    TARGET->sizeY   = INPUT.sizeY;
    TARGET->element = realloc(TARGET->element, sizeof(double) * size);

    for(; i < size; i++) TARGET->element[i] = sigmoid(INPUT.element[i]);
}

void sigmoid_dx_matrix(matrix *TARGET, matrix INPUT) {
    size_t size = INPUT.sizeX * INPUT.sizeY, i = 0;

    TARGET->sizeX   = INPUT.sizeX;
    TARGET->sizeY   = INPUT.sizeY;
    TARGET->element = realloc(TARGET->element, sizeof(double) * size);

    for(; i < size; i++) TARGET->element[i] = sigmoid_dx(INPUT.element[i]);
}

void add_matrix(matrix *TARGET, matrix A, matrix B) {
    size_t size = A.sizeX * A.sizeY, i = 0;

    TARGET->sizeX   = A.sizeX;
    TARGET->sizeY   = A.sizeY;
    TARGET->element = realloc(TARGET->element, sizeof(double) * size);

    for(; i < size; i++) TARGET->element[i] = A.element[i] + B.element[i];
}

void mul_matrix_alt(matrix *TARGET, matrix A, matrix B) {
    size_t size = A.sizeX * A.sizeY, i = 0;

    TARGET->sizeX   = A.sizeX;
    TARGET->sizeY   = A.sizeY;
    TARGET->element = realloc(TARGET->element, sizeof(double) * size);

    for(; i < size; i++) TARGET->element[i] = A.element[i] * B.element[i];
}

void sub_matrix(matrix *TARGET, matrix A, matrix B) {
    size_t size = A.sizeX * A.sizeY, i = 0;

    TARGET->sizeX   = A.sizeX;
    TARGET->sizeY   = A.sizeY;
    TARGET->element = realloc(TARGET->element, sizeof(double) * size);

    for(; i < size; i++) TARGET->element[i] = A.element[i] - B.element[i];
}

void mul_matrix(matrix *TARGET, matrix A, matrix B) {
    double MA[A.sizeX * A.sizeY], *MA_ptr = MA,
	   MB[B.sizeX * B.sizeY], *MB_ptr = MB;
    int i;
    for(i = 0; i < A.sizeX * A.sizeY; i++) MA[i] = A.element[i];
    for(i = 0; i < B.sizeX * B.sizeY; i++) MB[i] = B.element[i];

    TARGET->sizeX   = A.sizeX;
    TARGET->sizeY   = B.sizeY;
    TARGET->element = realloc(TARGET->element, sizeof(double) * A.sizeX * B.sizeY);
    double *TARGET_ptr = TARGET->element,
	   buffer[A.sizeX][B.sizeY];

    for(int IAx = 0; IAx < A.sizeX; IAx++) { 
	for(int IBy = 0; IBy < B.sizeY; IBy++) {
	    buffer[IAx][IBy] = 0;
	    for(int IAy = 0; IAy < A.sizeY; IAy++) {
		buffer[IAx][IBy] += *MA_ptr++ * *MB_ptr;
		MB_ptr += B.sizeY;
	    }
	    MA_ptr -= A.sizeY;
	    MB_ptr -= A.sizeY * B.sizeY - 1;

	    *TARGET_ptr++ = buffer[IAx][IBy];
	}
	MA_ptr += A.sizeY;
	MB_ptr -= B.sizeY;
    }
}

void mirror_matrix(matrix *TARGET, matrix INPUT) {
    double  buff[INPUT.sizeY][INPUT.sizeX],
	    *INPUT_ptr = INPUT.element;

    int ix, iy;
    for(ix = 0; ix < INPUT.sizeX; ix++)
	for(iy = 0; iy < INPUT.sizeY; iy++)
	     buff[iy][ix] = *INPUT_ptr++;
    
    TARGET->sizeX   = INPUT.sizeY;
    TARGET->sizeY   = INPUT.sizeX;
    TARGET->element = realloc(TARGET->element, sizeof(double) * INPUT.sizeX * INPUT.sizeY);
    double *TARGET_ptr = TARGET->element;

    for(iy = 0; iy < INPUT.sizeY; iy++)
	for(ix = 0; ix < INPUT.sizeX; ix++)
	    *TARGET_ptr++ = buff[iy][ix];
}

double dot_array(double *A, double *B, size_t size) {
    double buff = 0;
    for (int i = 0; i < size; i++) buff += A[i] * B[i];
    return buff;
}

void MOD_dot_matrix(matrix *TARGET, matrix A, matrix B) {
    double  A_buff[A.sizeX][A.sizeY], B_buff[B.sizeX][B.sizeY],
	    *A_ptr = A.element,
	    *B_ptr = B.element;

    int iA, iB, iY;

    for(iA = 0; iA < A.sizeX; iA++)
	for(iY = 0; iY < A.sizeY; iY++)
	    A_buff[iA][iY] = *A_ptr++;

    for(iB = 0; iB < B.sizeX; iB++)
	for(iY = 0; iY < B.sizeY; iY++)
	    B_buff[iB][iY] = *B_ptr++;

    TARGET->sizeX   = A.sizeX;
    TARGET->sizeY   = B.sizeX;
    TARGET->element = realloc(TARGET->element, sizeof(double) * A.sizeX * B.sizeX);
    double *TARGET_ptr = TARGET->element;

    for(iA = 0; iA < A.sizeX; iA++)
	for(iB = 0; iB < B.sizeX; iB++)
	    *TARGET_ptr++ = dot_array(A_buff[iA], B_buff[iB], A.sizeY);
}

void read_matrix(matrix *TARGET, char *filename) {
    FILE *INfile = fopen(filename, "r");
    char *buff   = malloc(sizeof(char)  * 255);
    int  sizeX = 1, sizeY = 0;

    while(buff[0] != ';') {
	fscanf(INfile, "%s", buff);
	if(buff[0] == ',') sizeX++;
	else if(sizeX == 1) sizeY++;
  }

    TARGET->sizeX   = sizeX;
    TARGET->sizeY   = sizeY;
    TARGET->element = realloc(TARGET->element, sizeof(double) * sizeX * sizeY);
    double *TARGET_ptr = TARGET->element;

    rewind(INfile);
    buff[0] = 'N';

    while(buff[0] != ';') {
	fscanf(INfile, "%s", buff);
	if(buff[0] != ',') *TARGET_ptr++ = atof(buff); 
    }

    free(buff);
    fclose(INfile);
}

void read_matrixs(matrix **TARGET, char *filename) {
    FILE *INfile = fopen(filename, "r");
    char *buff   = malloc(sizeof(char)  * 255);
    int matrix_amount = 0;

    while(strcmp(buff, "END")) {
	fscanf(INfile, "%s", buff);
	if(buff[0] == ';') matrix_amount++;
    }
    rewind(INfile);

    *TARGET = realloc(*TARGET, sizeof(matrix) * matrix_amount);
    inits(*TARGET, matrix_amount);
    TARGET[0]->number = matrix_amount;

    int X[matrix_amount], Y[matrix_amount], i = 0;
    for(; i < matrix_amount; i++) {
	X[i] = 1;
	Y[i] = 0;
	for(; buff[0] != ';';) {
	    fscanf(INfile, "%s", buff);
	    if(buff[0] == ',') X[i]++;
	    else if(X[i] == 1) Y[i]++;
	}
	buff[0] = 'N';
    }
    rewind(INfile);

    double *TARGET_ptr;
    for(i = 0; i < matrix_amount; i++) {
	TARGET[0]->sizeX   = X[i];
	TARGET[0]->sizeY   = Y[i];
	TARGET[0]->element = realloc(TARGET[0]->element, sizeof(double) * X[i] * Y[i]);
	TARGET_ptr = TARGET[0]->element;
	TARGET[0]++;

	for(; buff[0] != ';';) {
	    fscanf(INfile, "%s", buff);
	    if(buff[0] != ',') *TARGET_ptr++ = atof(buff);
	}
	buff[0] = 'N';
    }
    TARGET[0] -= matrix_amount;

    free(buff);
    fclose(INfile);
}

void write_matrix(matrix meta, char *filename) {
    FILE *file = fopen(filename, "w+");
    char write_buff[4096] , buff[512];

    for(int i = 0; i < meta.sizeX * meta.sizeY; i++) {
	if(meta.element[i] >= 0) sprintf(buff, " %.2f\t", meta.element[i]);
	else sprintf(buff, "%.2f\t", meta.element[i]);
	strcat(write_buff, buff);

	if((i + 1) % (meta.sizeX * meta.sizeY) == 0) strcat(write_buff, ";\n");
	else if((i + 1) % meta.sizeY == 0) strcat(write_buff, ",\n");
    }

    fputs(write_buff, file);
    fclose(file);
}

void write_matrixs(matrix *meta, char *filename) {
    int write_size = 0, meta_num = 0;
    for(; meta_num++ < meta[0].number; write_size += sizeof(char *) * meta[meta_num].sizeX * meta[meta_num].sizeY * 8) ;

    FILE *file = fopen(filename, "w");
    char write_buff[write_size], buff[512];

    for(meta_num = 0; meta_num < meta[0].number; meta_num++)
	for(int position = 0; position < meta[meta_num].sizeX * meta[meta_num].sizeY; position++) {
	    if(meta[meta_num].element[position] >= 0) sprintf(buff, " %.2f\t", meta[meta_num].element[position]);
	    else sprintf(buff, "%.2f\t", meta[meta_num].element[position]);
	    strcat(write_buff, buff);

	    if((position + 1) % (meta[meta_num].sizeX * meta[meta_num].sizeY) == 0) strcat(write_buff, ";\n");
	    else if((position + 1) % meta[meta_num].sizeY == 0) strcat(write_buff, ",\n");
	}

    strcat(write_buff, "END");
    fputs(write_buff, file);
    fclose(file);
}

#endif
