#include <stdio.h>
#include <stdint.h>

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

void readPnm(char * fileName, 
		int &numChannels, int &width, int &height, uint8_t * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P2") == 0)
		numChannels = 1;
	else if (strcmp(type, "P3") == 0)
		numChannels = 3;
	else // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uint8_t *)malloc(width * height * numChannels);
	for (int i = 0; i < width * height * numChannels; i++)
		fscanf(f, "%hhu", &pixels[i]);

	fclose(f);
}

void writePnm(uint8_t * pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}

struct pair_int_int {
    	int first;
    	int second;
};

int d[3] = {-1,0,1};

void convertRgb2Gray(uint8_t * inPixels, int width, int height,
		uint8_t * outPixels, 
		bool useDevice=false, dim3 blockSize=dim3(1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
        // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                int i = r * width + c;
                uint8_t red = inPixels[3 * i];
                uint8_t green = inPixels[3 * i + 1];
                uint8_t blue = inPixels[3 * i + 2];
                outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
            }
        }
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n\n", 
			useDevice == true? "use device" : "use host", time);
}

void apply_kernel(uint8_t * inPixels, int width, int height, float * filter, int filterWidth, 
		int * outPixels)
{
	for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
				// filter with convolution
				int count = 0;
				float res = 0;
                for (int r_filter = -filterWidth / 2; r_filter <= filterWidth/2; r_filter++){
					for (int c_filter = -filterWidth / 2; c_filter <= filterWidth/2; c_filter++) {
						int r_new = min(max(r + r_filter, 0),height-1);
						int c_new = min(max(c + c_filter, 0),width-1);
						int index = r_new * width + c_new;
						res += filter[count] * (float)inPixels[index];
						count++;
					}
				}
				int i = r * width + c;
				outPixels[i] = res;
            }
        }
}

void cal_important_pixel(int * inPixels_1 ,int * &inPixels_2, int * outPixels,int width, int height)
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
						outMatrixTrace[r*width + c] = d[k];
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

void get_trace(int *important_matrix_trace, int position,int width, int height, pair_int_int *res)
{
	int tmp_height = height, tmp_position = position;
	while (tmp_height--){
		res[tmp_height] = {tmp_height, tmp_position};
		// printf("%i %i %i\n",res[i].first, res[i].second,important_matrix_trace[tmp_height*width+tmp_position]);
		tmp_position += important_matrix_trace[tmp_height*width+tmp_position];
	}
}

void get_k_best(int * important_matrix, int * important_matrix_trace, 
				int width,int height, int k, pair_int_int * k_best)
{
	pair_int_int * tmp_list = (pair_int_int *)malloc(width *sizeof(pair_int_int));
	for (int i=0; i < width; i++)
	{
		tmp_list[i].first = important_matrix[(height-1) * width + i];
		tmp_list[i].second = i;
	}
	qsort(tmp_list, width, sizeof(pair_int_int),compare);

	for (int i=0; i<k; i++){
		get_trace(important_matrix_trace,tmp_list[i].second,width, height,k_best+i*height);
	}
	
}

char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

int main(int argc, char ** argv)
{	
	
	int numChannels, width, height;
	uint8_t * inPixels;
	readPnm(argv[1], numChannels, width, height, inPixels);
	if (numChannels != 3)
		return EXIT_FAILURE; // Input image must be RGB
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// Convert RGB to grayscale not using device
	uint8_t * grayscalePixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
	int * applyKernelPixels_1 = (int *)malloc(width * height * sizeof(int));
	int * applyKernelPixels_2 = (int *)malloc(width * height * sizeof(int));
	int * important_pixels = (int *)malloc(width * height * sizeof(int));
	int * important_matrix = (int *)malloc(width * height * sizeof(int));
	int * important_matrix_trace = (int *)malloc(width * height * sizeof(int));
	convertRgb2Gray(inPixels, width, height, grayscalePixels);

	int filterWidth = 3;
	// float * filter1 = (float *)malloc(filterWidth * filterWidth * sizeof(float));
	// float * filter2 = (float *)malloc(filterWidth * filterWidth * sizeof(float));
	float filter1[9] = {1,0,-1,2,0,-2,1,0,-1};
	float filter2[9] = {1,2,1,0,0,0,-1,-2,-1};

	apply_kernel(grayscalePixels, width, height, filter1, filterWidth, applyKernelPixels_1);
	apply_kernel(grayscalePixels, width, height, filter2, filterWidth, applyKernelPixels_2);
	cal_important_pixel(applyKernelPixels_1,applyKernelPixels_2,important_pixels,width, height);
	create_important_matrix(important_pixels,width, height,important_matrix,important_matrix_trace);
	int k = 2;
	pair_int_int * k_best_list = (pair_int_int *)malloc(k * height * sizeof(pair_int_int));
	get_k_best(important_matrix,important_matrix_trace, width, height, k, k_best_list);
	for (int r = 0; r < k; r++) {
        for (int c = 0; c < height; c++)
			printf("%i: %i %i \n",c, k_best_list[r*height + c].first, k_best_list[r*height+c].second);
		printf("\n");
	}

	// Write results to files
	// char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	// writePnm(applyKernelPixels, 1, width, height, concatStr(outFileNameBase, "_host.pnm"));

	// Free memories
	// free(inPixels);
}
