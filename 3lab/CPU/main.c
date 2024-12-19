#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double** MemMatrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

void FreeMatrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void GenerateMatrix(double** matrix, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = ((double)rand()/(double)(RAND_MAX)) * 21.764382;
        }
    }
}

void PrintMatrix(double** matrix, int n) {
    printf("Matrix %dx%d:\n", n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2lf ", matrix[i][j]);
        }
        printf("\n");
    }
}

void TriMatrix(double** matrix, int n) {
	for (int diag = 0; diag < n - 1; diag++) {
		for (int i = diag + 1; i < n; i++) {
			double k = (-1.0) * matrix[i][diag] / matrix[diag][diag];
			for (int j = diag; j < n; j++) {
				matrix[i][j] += k * matrix[diag][j];
			}
		}
	}
}

double Det(double** matrix, int n) {
    double det = 1.0;
    for (int i = 0; i < n; i++) {
        det *= matrix[i][i];
    }
    return det;
}

int main() {
    int n;
    double** matrix;

    printf("Matrix rank:\n");
    scanf("%d", &n);

    matrix = MemMatrix(n);
    GenerateMatrix(matrix, n);

    PrintMatrix(matrix, n);

    clock_t start = clock();

    TriMatrix(matrix, n);

    double det = Det(matrix, n);

    clock_t end = clock();

    printf("Matrix determinator: %.6lf\n", det);
    printf("Time to calc: %.6lf msec\n", (double)(end - start) / (CLOCKS_PER_SEC/1000));

    FreeMatrix(matrix, n);
    return 0;
}
