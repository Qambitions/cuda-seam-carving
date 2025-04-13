# CUDA Seam Carving

This project implements **seam carving**, a content-aware image resizing technique, accelerated using NVIDIA CUDA for enhanced performance.

## 🚀 Features

-Content-aware image resizing that preserves important visual content
-GPU acceleration with CUDA for improved processing speed
-Implemented in a Jupyter Notebook for interactive exploration

## 🧠 Algorithm Overview
Seam carving works by identifying and removing seams—paths of least importance—either vertically or horizontally, to reduce image dimensions while maintaining key visual feature.
The process involve:

- 1 Converting the image to grayscal.
- 2 Calculating an energy map using gradient magnitude (e.g., via Sobel filters.
- 3 Identifying the seam with the lowest cumulative energ.
- 4 Adding/Removing the identified seam from the imag.
- 5 Repeating the process until the desired image size is achieve.

By leveraging CUDA, the computation of energy maps and seam identification is parallelized, significantly reducing processing time compared to CPU-only implementation.

## 📁 Repository Structure

- `seam_carving.ipynb: Main notebook demonstrating the seam carving process with CUDA acceleratin.
- `cuda_code/: Contains CUDA source files and related utilitis.

## 🖼️ Example

*Original Image:*

![Original Image](img/hd.jpg)

*Resized Image:*

![Resized Image](img/hd_output.png)
