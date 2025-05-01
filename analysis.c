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

// SIMD-accelerated coordinate conversion
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

// SIMD-accelerated Euclidean distance
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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int data_size = 1000;
    double *latitudes = (double *)malloc(data_size * sizeof(double));
    double *longitudes = (double *)malloc(data_size * sizeof(double));
    int *lat_grid = (int *)malloc(data_size * sizeof(int));
    int *long_grid = (int *)malloc(data_size * sizeof(int));
    float *distances = (float *)malloc(data_size * data_size * sizeof(float));

    if (!latitudes || !longitudes || !lat_grid || !long_grid || !distances) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Generate synthetic test data
    for (int i = 0; i < data_size; i++) {
        latitudes[i] = min_lat + (max_lat - min_lat) * ((double)rand() / RAND_MAX);
        longitudes[i] = min_long + (max_long - min_long) * ((double)rand() / RAND_MAX);
    }

    // Coordinate conversion using SIMD
    lat_long_to_grid_simd(latitudes, longitudes, lat_grid, long_grid, data_size);

    // Distance computation using SIMD
    calculate_distances_simd(latitudes, longitudes, distances, data_size);

    // Spatial domain decomposition using MPI
    int chunk_size = data_size / nprocs;
    int start = rank * chunk_size;
    int end = (rank == nprocs - 1) ? data_size : (rank + 1) * chunk_size;

    int *local_counts = (int *)malloc(chunk_size * sizeof(int));
    for (int i = start; i < end; i++) {
        local_counts[i - start] = 0;
        for (int j = 0; j < data_size; j++) {
            if (lat_grid[i] == lat_grid[j] && long_grid[i] == long_grid[j]) {
                local_counts[i - start]++;
            }
        }
    }

    int *global_counts = NULL;
    if (rank == 0) {
        global_counts = (int *)malloc(data_size * sizeof(int));
    }

    MPI_Gather(local_counts, chunk_size, MPI_INT, global_counts, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global counts (first 10):\n");
        for (int i = 0; i < 10; i++) {
            printf("Cell %d: %d tweets\n", i, global_counts[i]);
        }
    }

    free(latitudes);
    free(longitudes);
    free(lat_grid);
    free(long_grid);
    free(distances);
    free(local_counts);
    if (rank == 0) free(global_counts);

    MPI_Finalize();
    return 0;
}
