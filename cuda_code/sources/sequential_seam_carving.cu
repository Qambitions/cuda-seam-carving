#pragma once
#include "./utils.cu"
#include <cuda_runtime.h>

/****************************************************************************/
/* IMPLEMENTATION OF SEQUENTIAL SEAM CARVING */
/****************************************************************************/
int d[3] = {-1,0,1};

void convert_rgb_to_grayscale(uchar3 * inPixels, int width, int height, int * outPixels)
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

// void apply_filter(uchar3* inPixels, int width, int height, float * filter, int filterWidth, uchar3* outPixels) {
void apply_filter(int* inPixels, int width, int height, float * filter, int filterWidth, int* outPixels) {
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
					fpixel += (float) inPixels[in_pos] * filter[f_pos];
				}
			}
			
			// fpixel = min(max(0.f, fpixel), 255.f);
			outPixels[pos] = (int) fpixel;
		}
	}
}

void calc_px_importance(int **inPixels, int* outPixels,int width, int height, int numFilters)
{
	for (int i = 0; i < height*width; i++) {
		// outPixels[i] = abs(inPixels_1[i])  + abs(inPixels_2[i]);	
		outPixels[i] = 0;
		for (int j = 0; j < numFilters; ++j)
			outPixels[i] += abs(inPixels[j][i]);
	}
}

void create_important_matrix(int * inPixels ,int width, int height, 
			int * outMatrix, int * outMatrixTrace)
{
	for (int r = 0; r < height; r++) 
        for (int c = 0; c < width; c++){ 
			outMatrix[r*width + c] = 1000000000;
			for (int k = 0; k < 3; k++)
				if (r > 0){
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
		// if (tmp_position < 0 || tmp_position >= width)
		// 	printf("%i %i %i %i %i\n", width, height, tmp_height, tmp_position, important_matrix_trace[tmp_height*width+tmp_position]);
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
		count += get_trace(important_matrix_trace,tmp_list[i].second,width, height,k_best+count*height);
	}
	free(tmp_list);
	return count;
}

void applyKSeams(uchar3* inPixels, uchar3* outPixels, int width, int height, pair_int_int* seams, int k, int mode) {
	if (mode == 0) {
		int outputWidth = width - k;
		// Reduce image size
		// Loop for each row of input image
		for (int i = 0; i < height; ++i) {
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
		}
	} else {
		int outputWidth = width + k;
		// Enlarge image size
		for (int i = 0; i < height; ++i) {
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