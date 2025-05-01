#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <mpi.h>

#define GRID_SIZE 0.01
#define EARTH_RADIUS_KM 6371.0

const double min_lat = 24.396308;
const double max_lat = 49.384358;
const double min_long = -125.000000;
const double max_long = -66.934570;

// File path for the CSV file (hardcoded)
const char *csv_filename = "/N/u/pkamburu/BigRed200/project/data/processed_twitter_data.csv";

// ------------------- Scalar coordinate conversion -------------------
void lat_long_to_grid_scalar(const double *latitudes, const double *longitudes, int *lat_grid, int *long_grid, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        lat_grid[i] = (int)((latitudes[i] - min_lat) / GRID_SIZE);
        long_grid[i] = (int)((longitudes[i] - min_long) / GRID_SIZE);
    }
}

// ------------------- SIMD coordinate conversion -------------------
void lat_long_to_grid_simd(const double *latitudes, const double *longitudes, int *lat_grid, int *long_grid, int size) {
    __m256d minLat = _mm256_set1_pd(min_lat);
    __m256d minLong = _mm256_set1_pd(min_long);
    __m256d cellSize = _mm256_set1_pd(GRID_SIZE);

    #pragma omp parallel for
    for (int i = 0; i < size; i += 4) {
        __m256d lat = _mm256_loadu_pd(&latitudes[i]);
        __m256d lon = _mm256_loadu_pd(&longitudes[i]);

        __m256d latIndex = _mm256_div_pd(_mm256_sub_pd(lat, minLat), cellSize);
        __m256d lonIndex = _mm256_div_pd(_mm256_sub_pd(lon, minLong), cellSize);

        double latIdx[4], lonIdx[4];
        _mm256_storeu_pd(latIdx, latIndex);
        _mm256_storeu_pd(lonIdx, lonIndex);

        for (int j = 0; j < 4 && (i + j) < size; j++) {
            lat_grid[i + j] = (int)latIdx[j];
            long_grid[i + j] = (int)lonIdx[j];
        }
    }
}

// ------------------- Scalar distance calculation -------------------
void calculate_distances_scalar(const double *lat, const double *lon, float *distances, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double dlat = lat[i] - lat[j];
            double dlon = lon[i] - lon[j];
            distances[i * size + j] = (float)sqrt(dlat * dlat + dlon * dlon);
        }
    }
}

// ------------------- SIMD distance calculation -------------------
void calculate_distances_simd(const double *lat, const double *lon, float *distances, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        __m256d lat_i = _mm256_set1_pd(lat[i]);
        __m256d lon_i = _mm256_set1_pd(lon[i]);
        for (int j = 0; j < size; j += 4) {
            __m256d lat_j = _mm256_loadu_pd(&lat[j]);
            __m256d lon_j = _mm256_loadu_pd(&lon[j]);

            __m256d dlat = _mm256_sub_pd(lat_i, lat_j);
            __m256d dlon = _mm256_sub_pd(lon_i, lon_j);

            __m256d dlat2 = _mm256_mul_pd(dlat, dlat);
            __m256d dlon2 = _mm256_mul_pd(dlon, dlon);
            __m256d dist2 = _mm256_add_pd(dlat2, dlon2);
            __m256d dist = _mm256_sqrt_pd(dist2);

            double dist_vals[4];
            _mm256_storeu_pd(dist_vals, dist);
            for (int k = 0; k < 4 && (j + k) < size; k++) {
                distances[i * size + j + k] = (float)dist_vals[k];
            }
        }
    }
}

// ------------------- Main -------------------
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // ------------------- Read CSV Data -------------------
    FILE *file = fopen(csv_filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", csv_filename);
        MPI_Finalize();
        return 1;
    }

    int data_size = 1000; // Change for larger tests
    double *latitudes = (double *)malloc(data_size * sizeof(double));
    double *longitudes = (double *)malloc(data_size * sizeof(double));
    int *lat_grid = (int *)malloc(data_size * sizeof(int));
    int *long_grid = (int *)malloc(data_size * sizeof(int));
    float *distances = (float *)malloc(data_size * data_size * sizeof(float));

    if (!latitudes || !longitudes || !lat_grid || !long_grid || !distances) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read the CSV file and populate latitudes and longitudes
    int i = 0;
    while (fscanf(file, "%lf,%lf", &latitudes[i], &longitudes[i]) == 2) {
        i++;
        if (i >= data_size) break; // Prevent overflow, increase data_size if needed
    }
    fclose(file);

    // ------------------- Timing Coordinate Conversion -------------------
    double t1 = omp_get_wtime();
    lat_long_to_grid_scalar(latitudes, longitudes, lat_grid, long_grid, data_size);
    double t2 = omp_get_wtime();

    double t3 = omp_get_wtime();
    lat_long_to_grid_simd(latitudes, longitudes, lat_grid, long_grid, data_size);
    double t4 = omp_get_wtime();

    // ------------------- Timing Distance Calculation -------------------
    double d1 = omp_get_wtime();
    calculate_distances_scalar(latitudes, longitudes, distances, data_size);
    double d2 = omp_get_wtime();

    double d3 = omp_get_wtime();
    calculate_distances_simd(latitudes, longitudes, distances, data_size);
    double d4 = omp_get_wtime();

    if (rank == 0) {
        printf("\n--- Coordinate Conversion ---\n");
        printf("Scalar time: %.6f seconds\n", t2 - t1);
        printf("SIMD time  : %.6f seconds\n", t4 - t3);
        printf("Speedup    : %.2fx\n", (t2 - t1) / (t4 - t3));

        printf("\n--- Distance Calculation ---\n");
        printf("Scalar time: %.6f seconds\n", d2 - d1);
        printf("SIMD time  : %.6f seconds\n", d4 - d3);
        printf("Speedup    : %.2fx\n", (d2 - d1) / (d4 - d3));
    }

    free(latitudes);
    free(longitudes);
    free(lat_grid);
    free(long_grid);
    free(distances);

    MPI_Finalize();
    return 0;
}

// Speed Analysi
// mpicc -O3 -mavx2 -fopenmp -o simd_hpc_project analysis.c -lm
// mpirun -np 1 ./simd_hpc_project