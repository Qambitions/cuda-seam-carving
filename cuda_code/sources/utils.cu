#pragma once
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

// Utility function to check for CUDA return errors
#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

// Utility structure to store seam info
struct pair_int_int {
    	int first;
    	int second;
};

// A class to count the elapsed running time
struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};


// Functions to read & write PNM image file
void readPnm(char * fileName, int &numChannels, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "rb");
	if (f == NULL) {
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}
	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P6") == 0) {
		numChannels = 3;
	} else
	{
		printf("PPM file flag is not P6\n"); 
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // We assume 1 byte per value
	{
		printf("Only 256-color (8 bit-per-channel-pixel) images are allowed\n");
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	fseek(f, 1, SEEK_CUR); // Handle redundant newline character when switching between scan and read
	fread(pixels, sizeof(uchar3), width * height, f);
	fclose(f);
}

void writePnm(void* pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "wb");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P6\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}
	fprintf(f, "%i %i\n255\n", width, height);
	if (numChannels == 1) {
		// fprintf(f, "\n");
		for (int i = 0; i < width * height; i++)
			fprintf(f, "%hhu\n", ((int*) pixels)[i]);
	} else if (numChannels == 3) {
		// fseek(f, -1, SEEK_CUR); // Handle trailing newline when switching from print to write
		fwrite(pixels, sizeof(uchar3), width * height, f);
	}
	fclose(f);
}

// Function to check if a string startwith a substring
bool StartsWith(const char *a, const char *b) {
   if(strncmp(a, b, strlen(b)) == 0) return 1;
   return 0;
}

// Utility function to color seams for debugging
void colorSeams(uchar3* inPixels, uchar3* outPixels, int width, int height, pair_int_int* seams, int k) {
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j)
			outPixels[i * width + j] = inPixels[i * width + j];
	}
	for (int i = 0; i < k * height; ++i) {
		int row = seams[i].first;
		int col = seams[i].second;
		outPixels[row * width + col].x = 255;
		outPixels[row * width + col].y = 0;
		outPixels[row * width + col].z = 0;
	}
}

// Utility function to check GPU info
void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu bytes\n", devProv.totalGlobalMem);
    printf("****************************\n\n");

}

// Comparators for sorting
int compare(const void *a, const void *b) {
  
    pair_int_int *pairA = (pair_int_int *)a;
    pair_int_int *pairB = (pair_int_int *)b;
  
    return pairA->first > pairB->first;
}

int compare_position(const void *a, const void *b) {
  
    pair_int_int *pairA = (pair_int_int *)a;
    pair_int_int *pairB = (pair_int_int *)b;
  
	return pairA->first > pairB->first || (pairA->first == pairB->first && pairA->second > pairB->second);
}