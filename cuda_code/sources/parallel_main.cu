#include "./parallel_seam_carving.cu"

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
	float avgTimes[] = {0,0,0,0,0,0,0};
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
		int *grayscalePixels = (int *)malloc(width * height * sizeof(int));
		
		timer.Start();
		convert_rgb_to_grayscale_cuda(inPixels, width, height, grayscalePixels, blockSize);
		timer.Stop();
		float time = timer.Elapsed();
		if (isVerbose)
			printf("Processing time: %f ms - Convert RGB to Grayscale\n", time);
		total_time_sequential += time;
		avgTimes[0] += time;

		if (loopTimes == 1) {
			char fName1[] = "grayscale.pnm";
			writePnm(grayscalePixels, 1, width, height, fName1);
		}

		// Do convolution with edge detection filters
		int* filteredPixels[4];
		for (int i = 0; i < 4; ++i)
			filteredPixels[i] = (int *)malloc(width * height * sizeof(int));

		timer.Start();
		apply_filter_cuda(grayscalePixels, width, height, filteredPixels, blockSize);	
		timer.Stop();
		time = timer.Elapsed();
		if (isVerbose)
			printf("Processing time: %f ms - Apply filters\n", time);
		total_time_sequential += time;
		avgTimes[1] += time;

		free(grayscalePixels); // Free grayscale matrix after done with it

		// Calculate importance of each pixel
		int * pixelImportance = (int *)malloc(width * height * sizeof(int));
		
		timer.Start();
		calc_px_importance(filteredPixels, pixelImportance, width, height, 4);
		timer.Stop();
		time = timer.Elapsed();
		if (isVerbose)
			printf("Processing time: %f ms - Calculate pixel importance\n", time);
		total_time_sequential += time;
		avgTimes[2] += time;
 
		for (int i = 0; i < 4; ++i)
			free(filteredPixels[i]); // Free filtered pixels after we're done with them

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
		avgTimes[3] += time;

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
		avgTimes[4] += time;

        // For debugging, output the seam visualization to a file
		// if ((loopTimes - 1) % 3 == 0) {
		// 	uchar3 *seamPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
		// 	colorSeams(inPixels, seamPixels, width, height, k_best_list, actualK);
		// 	char *fName = (char*)malloc(sizeof(char) * 20);
		// 	sprintf(fName, "seam_loop_%i.pnm", loopTimes);
		// 	writePnm(seamPixels, 3, width, height, fName);
		// 	free(seamPixels);
		// 	free(fName);
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
		avgTimes[5] += time;

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
		avgTimes[6] += time;

		free(k_best_list); // Free seam list after finishing

		// Prepare for next loop
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
	printf("-> Apply filters: %f ms\n", avgTimes[1] / loopTimes);
	printf("-> Calculate pixel importance: %f ms\n", avgTimes[2] / loopTimes);
	printf("-> Construct least pixel-importance matrix: %f ms\n", avgTimes[3] / loopTimes);
	printf("-> Find K least important seams: %f ms\n", avgTimes[4] / loopTimes);
	printf("-> Sort K seams' positions: %f ms\n", avgTimes[5] / loopTimes);
	printf("-> Reduce/Enlarge image: %f ms\n\n", avgTimes[6] / loopTimes);

	// Free memories
	free(inPixels);
	free(outPixels);
}