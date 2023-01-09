#include "./utils.cu"
#include <cuda_runtime.h>

/****************************************************************************/
/* IMPLEMENTATION OF SEQUENTIAL SEAM CARVING */
/****************************************************************************/
struct pair_int_int {
    	int first;
    	int second;
};

int d[3] = {-1,0,1};

void convert_rgb_to_grayscale(uchar3 * inPixels, int width, int height, uint8_t * outPixels)
{
	// Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			int i = r * width + c;
			uint8_t red = inPixels[i].x;
			uint8_t green = inPixels[i].y;
			uint8_t blue = inPixels[i].z;
			outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
		}
	}
	
}

void apply_filter_(uint8_t* inPixels, int width, int height, float * filter, int filterWidth, uint8_t* outPixels) {
	int half_fWidth = filterWidth / 2;
	// Loop over image
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			// Set output element to default value
			int pos = r * width + c;
			float fpixel = 0; //Use float to avoiding rounding error during summation

			// Loop over filter
			for (int f_r = -half_fWidth; f_r <= half_fWidth; ++f_r) {
				for (int f_c = -half_fWidth; f_c <= half_fWidth; ++f_c) {
					// Get the matrix element corresponding to the filter element's position
					int i = r + f_r,
						j = c + f_c;
					// Clamp the row and column indices if they are out of bounds
					i = i < 0 ? 0 : i;
					i = i > height - 1 ? height - 1 : i;
					j = j < 0 ? 0 : j;
					j = j > width - 1 ? width - 1 : j;
					// Calculate input position and filter position in 1D
					int in_pos = i * width + j,
						f_pos = (f_r + half_fWidth) * filterWidth + (f_c + half_fWidth);
					// Do convolution
					fpixel += inPixels[in_pos] * filter[f_pos];
				}
			}

			outPixels[pos] = (uint8_t) fpixel;
		}
	}
}

// void apply_filter(uint8_t * inPixels, int width, int height, float * filter, int filterWidth, int * outPixels)
// {
// 	for (int r = 0; r < height; r++) {
//             for (int c = 0; c < width; c++) {
// 				// filter with convolution
// 				int count = 0;
// 				float res = 0;
//                 for (int r_filter = -filterWidth / 2; r_filter <= filterWidth/2; r_filter++){
// 					for (int c_filter = -filterWidth / 2; c_filter <= filterWidth/2; c_filter++) {
// 						int r_new = min(max(r + r_filter, 0),height-1);
// 						int c_new = min(max(c + c_filter, 0),width-1);
// 						int index = r_new * width + c_new;
// 						res += filter[count] * (float)inPixels[index];
// 						count++;
// 					}
// 				}
// 				int i = r * width + c;
// 				outPixels[i] = res;
//             }
//         }
// }

void calc_px_importance(uint8_t *inPixels_1 , uint8_t *inPixels_2, int* outPixels,int width, int height)
{
	for (int i = 0; i < height*width; i++) 
		outPixels[i] = abs(inPixels_1[i])  + abs(inPixels_2[i]);	
}

void create_important_matrix(int * inPixels ,int width, int height, 
			int * outMatrix, int * outMatrixTrace)
{
	for (int r = 0; r < height; r++) 
        for (int c = 0; c < width; c++){ 
			outMatrix[r*width + c] = 1000000000;
			for (int k = 0; k < 3; k++)
				if (r > 0){
					int tmp = outMatrix[(r-1)*width + c+d[k]] + inPixels[r*width + c];
					if (0 <= c+d[k] && c+d[k] < width && 
						outMatrix[r*width + c] > tmp){
						outMatrix[r*width + c] = tmp;
						outMatrixTrace[r*width + c] = k;
					}
				}
				else
				{
					outMatrix[r*width + c] = inPixels[r*width + c];
					outMatrixTrace[r*width + c] = -1;
				}
		}
}

int compare(const void *a, const void *b) {
  
    pair_int_int *pairA = (pair_int_int *)a;
    pair_int_int *pairB = (pair_int_int *)b;
  
    return pairA->first > pairB->first;
}

int compare_position(const void *a, const void *b) {
  
    pair_int_int *pairA = (pair_int_int *)a;
    pair_int_int *pairB = (pair_int_int *)b;
  
    return pairA->first == pairB->first ? pairA->second < pairB->second : pairA->first < pairA->first;
}

int get_trace(int *important_matrix_trace, int position,int width, int height, pair_int_int *res)
{
	int tmp_height = height, tmp_position = position;
	int tmp_position_old = position;

	while (tmp_height--){
		printf("x%i ", tmp_height);
		int count = 0;
		if (tmp_height==0) break;
		while (important_matrix_trace[tmp_height*width+tmp_position] == -1){
			if (count == 3) return 0;
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

int get_k_best(int * important_matrix, int * important_matrix_trace, 
				int width,int height, int k, pair_int_int * k_best)
{
	pair_int_int * tmp_list = (pair_int_int *)malloc(width *sizeof(pair_int_int));
	for (int i=0; i < width; i++)
	{
		tmp_list[i].first = important_matrix[(height-1) * width + i];
		tmp_list[i].second = i;
	}
	qsort(tmp_list, width, sizeof(pair_int_int),compare);

	int count = 0;
	for (int i=0; i<width && count<k; i++){
		// get trace không thể song song
		count += get_trace(important_matrix_trace,tmp_list[i].second,width, height,k_best+count*height);
		printf("%i ", count);
	}
	return count;
}

__global__ void dp_cuda(int * inPixels ,int width, int height, int r, 
			int * outMatrix, int * outMatrixTrace)
{
	int d[3] = {-1,0,1};
	
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	outMatrix[r*width + c] = 1000000000;
	for (int k = 0; k < 3; k++)
		if (r > 0){
			int tmp = outMatrix[(r-1)*width + c+d[k]] + inPixels[r*width + c];
			if (0 <= c+d[k] && c+d[k] < width && 
				outMatrix[r*width + c] > tmp){
				outMatrix[r*width + c] = tmp;
				outMatrixTrace[r*width + c] = k;
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
	cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(tmp_list, out_pair, pair_nBytes, cudaMemcpyDeviceToHost));

	// for (int i=0; i < width; i++)
	// {
	// 	tmp_list[i].first = important_matrix[(height-1) * width + i];
	// 	tmp_list[i].second = i;
	// }
	
	qsort(tmp_list, width, sizeof(pair_int_int),compare);

	// số lượng K quá nhỏ để nên làm song song
	int count = 0;
	for (int i=0; i<width && count<k; i++){
		// get trace không thể song song
		count += get_trace(important_matrix_trace,tmp_list[i].second,width, height,k_best+count*height);
		// printf("%i ", count);
	}
	return count;
}

void create_important_matrix_cuda(int * important_pixels ,int width, int height, 
			int * outMatrix, int * outMatrixTrace, int blockSize){
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
}

void applyKSeams(uchar3* inPixels, uchar3* outPixels, int width, int height, pair_int_int* seams, int k, int mode) {
	if (mode == 0) {
		// Reduce image size
		// Loop for each row of input image
		for (int i = 0; i < height; ++i) {
			// Use 2 pointers to remove seams in a row
			int outIte = 0, inIte = 0, seamIte = 0;
			while (inIte < width) {
				if (seamIte >= k || inIte != seams[seamIte * height + i].second) {
					outPixels[outIte] = inPixels[inIte];
					++outIte;
				} else {
					inIte++;
					seamIte++;
				}
			}
		}
	} else {
		// Enlarge image size
		for (int i = 0; i < height; ++i) {
			// Use 2 pointers to duplicate seams in a row
			int outIte = width + k - 1, inIte = width - 1, seamIte = k - 1;
			while (inIte >= 0) {
				outPixels[outIte] = inPixels[inIte];
				--outIte;
				if (seamIte >= 0 && inIte == seams[seamIte * height + i].second) {
					outPixels[outIte] = inPixels[inIte];
					--outIte;
				}
				--inIte;
			}
		}
	}
	// // Copy the applied pixels to output
	// for (int i = 0 ; i < height; ++i) {
	// 	int row = i * desiredWidth;
	// 	for (int j = 0; j < desiredWidth; ++j) {
	// 		int index = row + j;
	// 		outPixels[index] = inPixels[index];
	// 	}
	// }
}

int main(int argc, char ** argv) {
    // Parse command-line arguments
    if (argc != 4 && argc != 6)
	{
		printf("Invalid run arguments.\nCommand: <executable> <path-to-input-PNM-image> <path-to-output-PNM-image> <desired-image-width> <max-seam-ratio> <cuda-block-size>\n");
		return EXIT_FAILURE;
	}
    char* inImg = argv[1];
    char* outImg = argv[2];
	int desiredWidth = atoi(argv[3]);
    int blockSize = 32;
	float maxSeamRatio = 0.5;
    if (argc >= 5)
        maxSeamRatio = atof(argv[4]);
	if (argc == 6)
		blockSize = atoi(argv[5]);
    printf("Run with block size: %d x %d - Max seam ratio: %.2f\n", blockSize, blockSize, maxSeamRatio);

    // Read input image
    int numChannels, width, height;
	uchar3 *inPixels;
	readPnm(inImg, numChannels, width, height, inPixels);
	if (numChannels != 3)
		return EXIT_FAILURE; // Input image must be RGB
	printf("Image size (width x height): %i x %i\n\n", width, height);
	printf("Desired image width: %d\n", desiredWidth);

	// Calculate number of seams needed
	int k = 0;
	int leftover = 0;
	int mode = 0; // Mode: 0 = reduce size, 1 = enlarge size
	if (desiredWidth < width) {
		k = width - desiredWidth;
	} else {
		k = desiredWidth - width;
		if (k > int(width / 2)) {
			leftover = k - int(width / 2);
			k = int(width / 2);
		}
		mode = 1;
	}

	// A variable to keep total run time
	float total_time_sequential = 0;

	// Convert RGB image to grayscale for easy processing
	uint8_t *grayscalePixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
	GpuTimer timer;
	
	timer.Start();
	convert_rgb_to_grayscale(inPixels, width, height, grayscalePixels);
	timer.Stop();
	float time = timer.Elapsed();
	printf("Convert RGB to Grayscale - Processing time: %f ms\n\n", time);
	total_time_sequential += time;

	// Do convolution with edge detection filters
	float filter1[9] = {1,0,-1,2,0,-2,1,0,-1}; // x-Sobel filter
	float filter2[9] = {1,2,1,0,0,0,-1,-2,-1}; // y-Sobel filter
	int filterWidth = 3;
	uint8_t * filteredPixels_1 = (uint8_t *)malloc(width * height * sizeof(uint8_t));
	uint8_t * filteredPixels_2 = (uint8_t *)malloc(width * height * sizeof(uint8_t));
	
	timer.Start();
	apply_filter_(grayscalePixels, width, height, filter1, filterWidth, filteredPixels_1);
	timer.Stop();
	time = timer.Elapsed();
	printf("Apply x-Sobel filter - Processing time: %f ms\n\n", time);
	total_time_sequential += time;

	timer.Start();
	apply_filter_(grayscalePixels, width, height, filter2, filterWidth, filteredPixels_2);
	timer.Stop();
	time = timer.Elapsed();
	printf("Apply y-Sobel filter - Processing time: %f ms\n\n", time);
	total_time_sequential += time;
	
	// Calculate importance of each pixel
	free(grayscalePixels); // Free grayscale matrix after done with it
	int * pixelImportance = (int *)malloc(width * height * sizeof(int));
	
	timer.Start();
	calc_px_importance(filteredPixels_1, filteredPixels_2, pixelImportance, width, height);
	timer.Stop();
	time = timer.Elapsed();
	printf("Calculate pixel importance - Processing time: %f ms\n\n", time);
	total_time_sequential += time;
	
	// Construct least pixel-importance matrix
	int * importantMatrix = (int *)malloc(width * height * sizeof(int));
	int * importantMatrixTrace = (int *)malloc(width * height * sizeof(int));

	timer.Start();
	create_important_matrix(pixelImportance, width, height, importantMatrix, importantMatrixTrace);
	timer.Stop();
	time = timer.Elapsed();
	printf("Construct least pixel-importance matrix - Processing time: %f ms\n\n", time);
	total_time_sequential += time;

	// Find K least important seams from the least pixel-importance matrix
	pair_int_int * k_best_list = (pair_int_int *)malloc(k * height * sizeof(pair_int_int));
	
	timer.Start();
	int actualK = get_k_best(importantMatrix, importantMatrixTrace, width, height, k, k_best_list);
	timer.Stop();
	time = timer.Elapsed();
	printf("Find K least important seams - Processing time: %f ms\n\n", time);
	total_time_sequential += time;
	printf("Needed %d seams. Actual seams found: %d\n\n", k, actualK);

	// Sort seam positions in each row for efficient remove/duplicate
	qsort(k_best_list, actualK * height, sizeof(pair_int_int), compare_position);

	// Remove or duplicate K seams to change image size
	uchar3 *outPixels = (uchar3 *)malloc(desiredWidth * height * sizeof(uchar3));

	timer.Start();
	applyKSeams(inPixels, outPixels, width, height, k_best_list, actualK, mode);
	timer.Stop();
	time = timer.Elapsed();
	printf("Reduce/Enlarge image - Processing time: %f ms\n\n", time);
	total_time_sequential += time;

	// Save output image
	writePnm(outPixels, 3, desiredWidth, height, outImg);

	// //############################################################
	// int * out_cuda = (int *)malloc(width * height * sizeof(int));
	// int * out_trace_cuda = (int *)malloc(width * height * sizeof(int));
	
	
	
	// get_k_best_cuda(important_matrix,important_matrix_trace, width, height, k, k_best_list,512);
	// for (int r = 0; r < k; r++) {
    //     for (int c = 0; c < height; c++)
	// 		printf("%i: %i %i \n",c, k_best_list[r*height + c].first, k_best_list[r*height+c].second);
	// 	printf("\n");
	// }

	// Write results to files
	// char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	// writePnm(applyKernelPixels, 1, width, height, concatStr(outFileNameBase, "_host.pnm"));

	// Output total run time
	printf("Total processing time: %f ms\n\n", total_time_sequential);

	// Free memories
	free(inPixels);
	free(filteredPixels_1);
	free(filteredPixels_2);
	free(pixelImportance);
	free(importantMatrix);
	free(importantMatrixTrace);
	free(k_best_list);
	free(outPixels);
}