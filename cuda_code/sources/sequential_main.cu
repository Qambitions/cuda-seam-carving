#include "./sequential_seam_carving.cu"

int main(int argc, char ** argv) {
    // Parse command-line arguments
    if (argc < 4 || argc > 7)
	{
		printf("Invalid run arguments.\nCommand: <executable> <path-to-input-PNM-image> <path-to-output-PNM-image> <desired-image-width> <is-verbose> <max-seam-ratio> <cuda-block-size>\n");
		return EXIT_FAILURE;
	}
    char* inImg = argv[1];
    char* outImg = argv[2];
	int desiredWidth = atoi(argv[3]);
    int blockSize = 32;
	float maxSeamRatio = 0.05;
	int isVerbose = 1;
	if (argc >= 5)
		isVerbose = atoi(argv[4]);
    if (argc >= 6)
        maxSeamRatio = atof(argv[5]);
	if (argc == 7)
		blockSize = atoi(argv[6]);
	if (desiredWidth < 0) {
		printf("Desired width can't be negative\n");
		return EXIT_FAILURE;
	}
	if (maxSeamRatio < 0 || maxSeamRatio > 1) {
		printf("Max seam ratio must be a floating-point number within range [0,1]\n");
		return EXIT_FAILURE;
	}
    printf("Run with block size: %d x %d - Max seam ratio: %.2f\n", blockSize, blockSize, maxSeamRatio);

    // Read input image
    int numChannels, width, height;
	uchar3 *inPixels = nullptr, *outPixels = nullptr;
	readPnm(inImg, numChannels, width, height, outPixels); // Read to outPixels so we can pass to inPixels later in the loop
	if (numChannels != 3)
		return EXIT_FAILURE; // Input image must be RGB
	printf("Image size (width x height): %i x %i\n", width, height);
	printf("Desired image width: %d\n", desiredWidth);

	// Calculate number of seams needed
	int seamNeeded = 0;
	int mode = 0; // Mode: 0 = reduce size, 1 = enlarge size
	if (desiredWidth < width) {
		seamNeeded = width - desiredWidth;
	} else {
		seamNeeded = desiredWidth - width;
		mode = 1;
	}
	int maxSeam = int(width * maxSeamRatio) > 1? int(width * maxSeamRatio) : 1;
	printf("Total seams needed: %i - Max usable seams: %i\n\n", seamNeeded, maxSeam);

	// Variables to keep total & avg run time
	float total_time_sequential = 0;
	float avgTimes[] = {0,0,0,0,0,0,0,0};
	GpuTimer timer;
	int loopTimes = 1;

	while (seamNeeded > 0) {
		if (isVerbose)
			printf("LOOP #%i\n\n", loopTimes);
		// Use output from previous loop as input
		if (inPixels != nullptr)
			free(inPixels);
		inPixels = outPixels;
		int seamUse = seamNeeded > maxSeam? maxSeam : seamNeeded;

		// Convert RGB image to grayscale for easy processing
		uint8_t *grayscalePixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
		
		timer.Start();
		convert_rgb_to_grayscale(inPixels, width, height, grayscalePixels);
		timer.Stop();
		float time = timer.Elapsed();
		if (isVerbose)
			printf("Processing time: %f ms - Convert RGB to Grayscale\n", time);
		total_time_sequential += time;
		avgTimes[0] += time;

		// char fName1[] = "grayscale.pnm";
		// writePnm(grayscalePixels, 1, width, height, fName1);

		// Do convolution with edge detection filters
		float filter1[9] = {1,0,-1,2,0,-2,1,0,-1}; // x-Sobel filter
		float filter2[9] = {1,2,1,0,0,0,-1,-2,-1}; // y-Sobel filter
		int filterWidth = 3;
		uint8_t * filteredPixels_1 = (uint8_t *)malloc(width * height * sizeof(uint8_t));
		uint8_t * filteredPixels_2 = (uint8_t *)malloc(width * height * sizeof(uint8_t));

		timer.Start();
		apply_filter(grayscalePixels, width, height, filter1, filterWidth, filteredPixels_1);
		timer.Stop();
		time = timer.Elapsed();
		if (isVerbose)
			printf("Processing time: %f ms - Apply x-Sobel filter\n", time);
		total_time_sequential += time;
		avgTimes[1] += time;

		// char fName2[] = "xsobel.pnm";
		// writePnm(filteredPixels_1, 1, width, height, fName2);

		timer.Start();
		apply_filter(grayscalePixels, width, height, filter2, filterWidth, filteredPixels_2);
		timer.Stop();
		time = timer.Elapsed();
		if (isVerbose)
			printf("Processing time: %f ms - Apply y-Sobel filter\n", time);
		total_time_sequential += time;
		avgTimes[2] += time;

		// char fName3[] = "ysobel.pnm";
		// writePnm(filteredPixels_2, 1, width, height, fName3);

		free(grayscalePixels); // Free grayscale matrix after done with it

		// Calculate importance of each pixel
		int * pixelImportance = (int *)malloc(width * height * sizeof(int));
		
		timer.Start();
		calc_px_importance(filteredPixels_1, filteredPixels_2, pixelImportance, width, height);
		timer.Stop();
		time = timer.Elapsed();
		if (isVerbose)
			printf("Processing time: %f ms - Calculate pixel importance\n", time);
		total_time_sequential += time;
		avgTimes[3] += time;

		free(filteredPixels_1); // Free filtered pixels after we're done with them
		free(filteredPixels_2);

		// Construct least pixel-importance matrix
		int * importantMatrix = (int *)malloc(width * height * sizeof(int));
		int * importantMatrixTrace = (int *)malloc(width * height * sizeof(int));

		timer.Start();
		create_important_matrix(pixelImportance, width, height, importantMatrix, importantMatrixTrace);
		timer.Stop();
		time = timer.Elapsed();
		if (isVerbose)
			printf("Processing time: %f ms - Construct least pixel-importance matrix\n", time);
		total_time_sequential += time;
		avgTimes[4] += time;

		free(pixelImportance); // Free pixel importance after we have the matrix

		// Find K least important seams from the least pixel-importance matrix
		pair_int_int * k_best_list = (pair_int_int *)malloc(seamUse * height * sizeof(pair_int_int));
		
		timer.Start();
		int actualK = get_k_best(importantMatrix, importantMatrixTrace, width, height, seamUse, k_best_list);
		timer.Stop();
		time = timer.Elapsed();
		if (isVerbose) {
			printf("Processing time: %f ms - Find K least important seams\n", time);
			printf("Needed %d seams. Actual seams found: %d\n", seamUse, actualK);
		}
		total_time_sequential += time;
		avgTimes[5] += time;

        // For debugging, output the seam visualization to a file
		// if ((loopTimes - 1) % 3 == 0) {
		// 	uchar3 *seamPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
		// 	colorSeams(inPixels, seamPixels, width, height, k_best_list, actualK);
		// 	char *fName = (char*)malloc(sizeof(char) * 20);
		// 	sprintf(fName, "seam_loop_%i.pnm", loopTimes);
		// 	writePnm(seamPixels, 3, width, height, fName);
		// 	free(seamPixels);
		// }

		free(importantMatrix); // Free the importance matrix after we're done with it
		free(importantMatrixTrace);

		// Sort seam positions in each row for efficient remove/duplicate
		timer.Start();
		qsort(k_best_list, actualK * height, sizeof(pair_int_int), compare_position);
		timer.Stop();
		time = timer.Elapsed();
		if (isVerbose)
			printf("Processing time: %f ms - Sort K seams' positions\n", time);
		total_time_sequential += time;
		avgTimes[6] += time;

		// Remove or duplicate K seams to change image size
		int outWidth = 0;
		if (mode == 0)
			outWidth = width - actualK;
		else outWidth = width + actualK;
		outPixels = (uchar3 *)malloc(outWidth * height * sizeof(uchar3));

		timer.Start();
		applyKSeams(inPixels, outPixels, width, height, k_best_list, actualK, mode);
		timer.Stop();
		time = timer.Elapsed();
		if (isVerbose)
			printf("Processing time: %f ms - Reduce/Enlarge image\n\n", time);
		total_time_sequential += time;
		avgTimes[7] += time;

		free(k_best_list); // Free seam list after finishing

		// Prepare for next loop
		// char *fName = (char*)malloc(sizeof(char) * 20);
		// sprintf(fName, "inter_%i.pnm", loopTimes);
		// writePnm(outPixels, 3, outWidth, height, fName);
		width = outWidth;
		seamNeeded -= actualK;
		loopTimes++;
	}

	// Save output image
	writePnm(outPixels, 3, desiredWidth, height, outImg);
	printf("Saved output image to '%s'\n\n", outImg);

	// Output total & avg run time
	printf("Total processing time: %f ms\n\n", total_time_sequential);
	printf("Average time in each phase:\n");
	printf("-> Convert RGB to Grayscale: %f ms\n", avgTimes[0] / loopTimes);
	printf("-> Apply x-Sobel filter: %f ms\n", avgTimes[1] / loopTimes);
	printf("-> Apply y-Sobel filter: %f ms\n", avgTimes[2] / loopTimes);
	printf("-> Calculate pixel importance: %f ms\n", avgTimes[3] / loopTimes);
	printf("-> Construct least pixel-importance matrix: %f ms\n", avgTimes[4] / loopTimes);
	printf("-> Find K least important seams: %f ms\n", avgTimes[5] / loopTimes);
	printf("-> Sort K seams' positions: %f ms\n", avgTimes[6] / loopTimes);
	printf("-> Reduce/Enlarge image: %f ms\n\n", avgTimes[7] / loopTimes);

	// Free memories
	free(inPixels);
	free(outPixels);
}