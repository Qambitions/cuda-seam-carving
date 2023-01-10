#pragma once
#include "./utils.cu"
#include <cuda_runtime.h>

/****************************************************************************/
/* IMPLEMENTATION OF PARALLEL SEAM CARVING */
/****************************************************************************/
const int NFILTERS = 4;

__constant__ int filterWidth = 3;

__constant__ float filters[4][9] = {
	// left-Sobel filter
	{1.0,0.0,-1.0,2.0,0.0,-2.0,1.0,0.0,-1.0},
	// right-Sobel filter
	{-1.0,0.0,1.0,-2.0,0.0,2.0,-1.0,0.0,1.0},
	// top-Sobel filter
	{1.0,2.0,1.0,0.0,0.0,0.0,-1.0,-2.0,-1.0},
	// outline filter
	{-1.0,-1.0,-1.0,-1.0,8.0,-1.0,-1.0,-1.0,-1.0}
};

int d[3] = {-1,0,1};

__global__ void rgb_to_grayscale_kernel(uchar3 * inPixels, int width, int height, int * outPixels) {
	// Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
	int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < width && r < height) {
		int i = r * width + c;
		uint8_t red = inPixels[i].x;
		uint8_t green = inPixels[i].y;
		uint8_t blue = inPixels[i].z;
		outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
	}
}

void convert_rgb_to_grayscale_cuda(uchar3 * inPixels, int width, int height, int * outPixels, int blockSize)
{
	// Allocate device memory
	uchar3 *d_inPixels;
	int *d_outPixels;
	size_t inBytes = width * height * sizeof(uchar3);
	size_t outBytes = width * height * sizeof(int);
	CHECK(cudaMalloc(&d_inPixels, inBytes));
	CHECK(cudaMalloc(&d_outPixels, outBytes));
	// Copy data to device memory
	CHECK(cudaMemcpy(d_inPixels, inPixels, inBytes, cudaMemcpyHostToDevice));
	// Set grid size and call kernel
	dim3 blkSize(blockSize, blockSize);
	dim3 gridSize((width - 1) / blockSize + 1, (height - 1) / blockSize + 1);
	rgb_to_grayscale_kernel<<<gridSize, blkSize>>>(d_inPixels, width, height, d_outPixels);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	// Copy result from device memory
	CHECK(cudaMemcpy(outPixels, d_outPixels, outBytes, cudaMemcpyDeviceToHost));
	// Free device memory
	CHECK(cudaFree(d_inPixels));
	CHECK(cudaFree(d_outPixels));
}

__global__ void filter_kernel(int* inPixels, int width, int height, int* outPixels, int streamIdx) {	
	int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ int s_inPixels[];

	if (r < height && c < width)
		s_inPixels[threadIdx.y * blockDim.x + threadIdx.x] = inPixels[r * width + c];

	__syncthreads(); // Wait for all threads in block to allocate shared memory for faster input read
	
	int half_fWidth = filterWidth / 2;
	if (r < height && c < width){
		int count = 0, index, r_new, c_new, r_out, c_out;
		float fpixel = 0.0;
		int pos = r * width + c;

		for (int r_filter = -half_fWidth; r_filter <= half_fWidth; r_filter++){
			for (int c_filter = -half_fWidth; c_filter <= half_fWidth; c_filter++) {
				r_new = threadIdx.y + r_filter;
				c_new = threadIdx.x + c_filter;

				if (0 <= r_new && r_new < blockDim.y &&
				    0 <= c_new && c_new < blockDim.x) {
					index = r_new * blockDim.x + c_new;
					fpixel += (float) s_inPixels[index] * filters[streamIdx][count];
				}
				else { 
					r_out = min(max(r + r_filter, 0),height-1);
					c_out = min(max(c + c_filter, 0),width-1);
					index = r_out * width + c_out;
					fpixel += (float) inPixels[index] * filters[streamIdx][count];
				}
				count++;
			}
		}

		// fpixel = min(max(0.f, fpixel), 255.f);
		outPixels[pos] = (int) fpixel;
	}

	// int r = blockIdx.y * blockDim.y + threadIdx.y;
    // int c = blockIdx.x * blockDim.x + threadIdx.x;

	// if (r < height && c < width){
	// 	int count = 0, index, r_new, c_new;
	// 	float fpixel = 0;
	// 	for (int r_filter = -filterWidth / 2; r_filter <= filterWidth/2; r_filter++){
	// 		for (int c_filter = -filterWidth / 2; c_filter <= filterWidth/2; c_filter++) {
	// 			r_new = min(max(r + r_filter, 0),height-1);
	// 			c_new = min(max(c + c_filter, 0),width-1);
	// 			index = r_new * width + c_new;
	// 			fpixel += filters[streamIdx][count] * (float) inPixels[index];
	// 			count++;
	// 		}
	// 	}
	// 	int i = r * width + c;
	// 	outPixels[i] = (int) fpixel;
	// }
}

__global__ void filter_kernel2(int* inPixels, int width, int height, int* outPixels, int streamIdx) {	
	int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

	if (r < height && c < width){
		int count = 0, index, r_new, c_new;
		float fpixel = 0;
		for (int r_filter = -filterWidth / 2; r_filter <= filterWidth/2; r_filter++){
			for (int c_filter = -filterWidth / 2; c_filter <= filterWidth/2; c_filter++) {
				r_new = min(max(r + r_filter, 0),height-1);
				c_new = min(max(c + c_filter, 0),width-1);
				index = r_new * width + c_new;
				fpixel += filters[streamIdx][count] * (float) inPixels[index];
				count++;
			}
		}
		int i = r * width + c;
		outPixels[i] = (int) fpixel;
	}
}

void apply_filter_cuda(int* inPixels, int width, int height, int** outPixels, int blockSize) {
	// Pin host memory for async memcpy
	size_t nBytes = width * height * sizeof(int);
	CHECK(cudaHostRegister(inPixels, nBytes, cudaHostRegisterDefault));
	for (int i = 0; i < NFILTERS; ++i) {
		CHECK(cudaHostRegister(outPixels[i], nBytes, cudaHostRegisterDefault));
	}

	// Allocate device memory to use with n-filter streams
	int *d_in, **d_out;
	CHECK(cudaMalloc(&d_in, nBytes));
	d_out = (int**) malloc(NFILTERS * sizeof(int*));
	for (int i = 0; i < NFILTERS; ++i) {
		CHECK(cudaMalloc(&d_out[i], nBytes));
	}

	// Create "nStreams" device streams
	cudaStream_t *streams = (cudaStream_t*) malloc(NFILTERS * sizeof(cudaStream_t));
	for (int i = 0; i < NFILTERS; ++i) {
		CHECK(cudaStreamCreate(&streams[i]));
	}

	// Let each stream performs convolution with each filter separately
	// Copy input to device
	CHECK(cudaMemcpy(d_in, inPixels, nBytes, cudaMemcpyHostToDevice));
	for (int i = 0; i < NFILTERS; ++i) {
		// Set grid size
		dim3 blkSize(blockSize, blockSize);
		dim3 gridSize((width - 1) / blockSize + 1, (height - 1) / blockSize + 1);
		filter_kernel<<<gridSize, blkSize, (blkSize.x + 1) * (blkSize.y + 1) * sizeof(int), streams[i]>>>(d_in, width, height, d_out[i], i);
		CHECK(cudaDeviceSynchronize());
		// Copy device output to host
		CHECK(cudaMemcpyAsync(outPixels[i], d_out[i], nBytes, cudaMemcpyDeviceToHost, streams[i]));
		CHECK(cudaDeviceSynchronize());
	}

	// Wait for all streams to complete
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());

	// Destroy device streams
	for (int i = 0; i < NFILTERS; ++i) {
		CHECK(cudaStreamDestroy(streams[i]));
	}
	free(streams);

	// Free device memory regions
	CHECK(cudaFree(d_in));
	for (int i = 0; i < NFILTERS; ++i) {
		CHECK(cudaFree(d_out[i]));
	}
	free(d_out);

	// Unpin host memory regions
	CHECK(cudaHostUnregister(inPixels));
	for (int i = 0; i < NFILTERS; ++i) {
		CHECK(cudaHostUnregister(outPixels[i]));
	}
}

__global__ void calc_important_kernel(int *inPixels, int* outPixels,int width, int height) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

	if (r < height && c < width) {
		int pos = r * width + c;
		outPixels[pos] = 0;
		for (int j = 0; j < NFILTERS; ++j)
			outPixels[pos] += abs(inPixels[j * width * height + pos]);
	}
}

void calc_px_importance_cuda(int **inPixels, int* outPixels,int width, int height, int blockSize)
{
	// Allocate device memory
	int *d_inPixels;
	int *d_outPixels;
	size_t nBytes = width * height * sizeof(int);
	CHECK(cudaMalloc(&d_inPixels, NFILTERS * nBytes));
	CHECK(cudaMalloc(&d_outPixels, nBytes));

	// Copy data to device memory
	for (int i = 0; i < NFILTERS; ++i)
		CHECK(cudaMemcpy(d_inPixels + i * width * height, inPixels[i], nBytes, cudaMemcpyHostToDevice));
	
	// Set grid size and call kernel
	dim3 blkSize(blockSize, blockSize);
	dim3 gridSize((width - 1) / blockSize + 1, (height - 1) / blockSize + 1);
	calc_important_kernel<<<gridSize, blkSize>>>(d_inPixels, d_outPixels, width, height);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	// Copy result from device memory
	CHECK(cudaMemcpy(outPixels, d_outPixels, nBytes, cudaMemcpyDeviceToHost));
	// Free device memory
	CHECK(cudaFree(d_inPixels));
	CHECK(cudaFree(d_outPixels));
}

int get_trace(int *important_matrix_trace, int position,int width, int height, pair_int_int *res)
{
	int tmp_height = height, tmp_position = position;
	int tmp_position_old = position;

	while (tmp_height--){
		int count = 0;
		if (tmp_height==0) break;
		while (important_matrix_trace[tmp_height*width+tmp_position] == -1){
			if (count == 3) return 0;
			if (tmp_position_old + d[count] >= 0 && tmp_position_old + d[count] < width)
				tmp_position = tmp_position_old + d[count];
			count += 1;
		}
		res[tmp_height] = {tmp_height, tmp_position};
		tmp_position_old = tmp_position;
		int tmp = d[important_matrix_trace[tmp_height*width+tmp_position]];
		important_matrix_trace[tmp_height*width+tmp_position] = -1;
		tmp_position += tmp;
		
	}
	res[tmp_height] = {tmp_height, tmp_position};
	return 1;
}

__global__ void dp_cuda(int * inPixels ,int width, int height, int r, 
			int * outMatrix, int * outMatrixTrace)
{
	int d[3] = {-1,0,1};
	
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	outMatrix[r*width + c] = 1000000000;
	for (int k = 0; k < 3; k++)
		if (r > 0) {
			if (0 <= c+d[k] && c+d[k] < width) {
				int tmp = outMatrix[(r-1)*width + c+d[k]] + inPixels[r*width + c];
				if (outMatrix[r*width + c] > tmp){
					outMatrix[r*width + c] = tmp;
					outMatrixTrace[r*width + c] = k;
				}
			}
		}
		else
		{
			outMatrix[r*width + c] = inPixels[r*width + c];
			outMatrixTrace[r*width + c] = -1;
		}
}

__global__ void create_pair(int * d_in, int width, int height, pair_int_int * out_pair){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<width){
		out_pair[i].first = d_in[i];
		out_pair[i].second = i;
	}
}

int get_k_best_cuda(int * important_matrix, int * important_matrix_trace, 
				int width,int height, int k, pair_int_int * k_best,int blockSize)
{
	// Find seam energy-position pairs in parallel
	size_t pair_nBytes = width * sizeof(pair_int_int);
  	size_t nBytes = width * sizeof(int);
	pair_int_int * tmp_list = (pair_int_int *)malloc(width *sizeof(pair_int_int));
	dim3 gridSize_x1((width - 1) / blockSize + 1);
	int * d_important_matrix;
	pair_int_int * out_pair;
	CHECK(cudaMalloc(&d_important_matrix, nBytes));
	CHECK(cudaMalloc(&out_pair, pair_nBytes));
	int index = (height-1)*width;
	CHECK(cudaMemcpy(d_important_matrix, important_matrix+index, nBytes, cudaMemcpyHostToDevice));
	create_pair<<<gridSize_x1, blockSize>>>(d_important_matrix,width, height, out_pair);
	CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(tmp_list, out_pair, pair_nBytes, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(d_important_matrix));
	CHECK(cudaFree(out_pair));

	// Traceback seams sequentially
	qsort(tmp_list, width, sizeof(pair_int_int),compare);
	
	int count = 0;
	for (int i=0; i<width && count<k; i++){
		// get_trace can't be parallelize
		count += get_trace(important_matrix_trace,tmp_list[i].second,width, height,k_best+count*height);
	}
	free(tmp_list);
	return count;
}

void create_important_matrix_cuda(int * important_pixels ,int width, int height, 
			int * outMatrix, int * outMatrixTrace, int blockSize = 512){
	// nice version
	size_t nBytes = width * height * sizeof(int);
	dim3 gridSize_x1((width - 1) / blockSize + 1);
	
	int * d_important_pixels, * d_important_matrix, * d_important_matrix_trace;
	CHECK(cudaMalloc(&d_important_pixels, nBytes));
	CHECK(cudaMalloc(&d_important_matrix, nBytes));
	CHECK(cudaMalloc(&d_important_matrix_trace, nBytes));
	CHECK(cudaMemcpy(d_important_pixels, important_pixels, nBytes, cudaMemcpyHostToDevice));
	for (int r=0; r<height; r++){
		dp_cuda<<<gridSize_x1, blockSize>>>(d_important_pixels,width, height, r, d_important_matrix,d_important_matrix_trace);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
	}
	CHECK(cudaMemcpy(outMatrix, d_important_matrix, nBytes, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(outMatrixTrace, d_important_matrix_trace, nBytes, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(d_important_pixels));
	CHECK(cudaFree(d_important_matrix));
	CHECK(cudaFree(d_important_matrix_trace));
}

__global__ void applyKSeams_kernel(uchar3* inPixels, uchar3* outPixels, int width, int height, pair_int_int* seams, int k, int mode) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < height) {
		if (mode == 0) {
			int outputWidth = width - k;
			// Reduce image size
			// Use 2 pointers to remove seams in a row
			int outIte = 0, inIte = 0, seamIte = 0;
			while (inIte < width) {
				if (seamIte >= k || inIte != seams[i * k + seamIte].second) {
					outPixels[i * outputWidth + outIte] = inPixels[i * width + inIte];
					++outIte;
				} else {
					++seamIte;
				}
				++inIte;
			}
		} else {
			int outputWidth = width + k;
			// Enlarge image size
			// Use 2 pointers to duplicate seams in a row
			int outIte = 0, inIte = 0, seamIte = 0;
			while (outIte < outputWidth) {
				outPixels[i * outputWidth + outIte] = inPixels[i * width + inIte];
				++outIte;
				while (outIte < outputWidth && seamIte < k && inIte == seams[i * k + seamIte].second) {
					outPixels[i * outputWidth + outIte] = inPixels[i * width + inIte];
					++outIte;
					++seamIte;
				}
				++inIte;
			}
		}
	}
}

void applyKSeams_cuda(uchar3* inPixels, uchar3* outPixels, int width, int height, pair_int_int* seams, int k, int mode, int blockSize) {
	// Allocate memory
	size_t nBytes = width * height * sizeof(uchar3);
	size_t nBytesOut = (width + k) * height * sizeof(uchar3);
	size_t seamNBytes = k * height * sizeof(pair_int_int);
	uchar3* d_in, *d_out;
	pair_int_int* d_seams;
	CHECK(cudaMalloc(&d_in, nBytes));
	CHECK(cudaMalloc(&d_out, nBytesOut));
	CHECK(cudaMalloc(&d_seams, seamNBytes));
	// Copy to device
	CHECK(cudaMemcpy(d_in, inPixels, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_seams, seams, seamNBytes, cudaMemcpyHostToDevice));
	// Call kernel
	dim3 blkSize(blockSize);
	dim3 gridSize((height - 1) / blockSize + 1);
	applyKSeams_kernel<<<gridSize, blkSize>>>(d_in, d_out, width, height, d_seams, k, mode);
	CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
	// Copy back to host
	CHECK(cudaMemcpy(outPixels, d_out, nBytesOut, cudaMemcpyDeviceToHost));
	// Free memory
	CHECK(cudaFree(d_in));
	CHECK(cudaFree(d_out));
	CHECK(cudaFree(d_seams));
}