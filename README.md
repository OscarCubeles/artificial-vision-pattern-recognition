# Artificial Vision and Pattern Recognition

This repository contains the coursework completed for the **Artificial Vision and Pattern Recognition** subject. It brings together the practical laboratory exercises, a larger image-processing project, and a research study based on a published computer-vision paper. The work covers classical image processing, local feature extraction and matching, deep image classification, semantic segmentation, and the analysis of a recent scene-recognition architecture.

## Labs

### Lab 1 - Image filtering and frequency-domain processing

Lab 1 introduces fundamental image-processing operations with OpenCV and NumPy. It compares Sobel, Prewitt, and Scharr edge detectors; explores custom sharpening, enhancement, embossing, and blur kernels; evaluates several denoising filters on salt-and-pepper noise using PSNR; simulates motion blur at different sizes and angles; and studies high-pass and low-pass filtering in the Fourier domain. The main implementation is available in [`lab1.ipynb`](labs/Lab1/lab1.ipynb), with the accompanying exercise document in [`LAB_1.pdf`](labs/Lab1/LAB_1.pdf).

### Lab 2 - Edges, corners, and morphology

Lab 2 examines structural feature detection and mathematical morphology. The notebook compares gradient-based and Canny edge detection, studies the effect of Canny thresholds, applies Harris corner detection with different parameter choices, and evaluates erosion, dilation, opening, closing, and morphological gradients on several types of images. The main implementation is [`lab2.ipynb`](labs/Lab2/lab2.ipynb); a Python export is also provided in [`lab2.py`](labs/Lab2/lab2.py), together with the [`lab report`](labs/Lab2/oscar.cubeles_lab2_AVPR.pdf).

### Lab 3 - Feature descriptors and local feature matching

Lab 3 focuses on representing and matching visual features. It extracts and spatially distributes SIFT keypoints, performs descriptor matching between images, computes Local Binary Pattern (LBP) texture descriptors, aligns images using matched features and estimated transformations, and trains a k-nearest-neighbors classifier for texture recognition. The main source code and analysis are contained in [`Lab3AVPR_oscar.cubeles.ipynb`](labs/Lab3/Lab3AVPR_oscar.cubeles.ipynb), with results documented in the [`lab report`](labs/Lab3/Lab3AVPR_oscar.cubeles.pdf).

### Lab 5 - ResNet training and experimentation

Lab 5 studies deep image classification with PyTorch and a custom ResNet. It investigates the effects of learning rate, batch size, and training duration; adapts the network architecture through dropout, convolutional kernel, filter, and depth changes; and compares data-transformation and augmentation strategies. Experimental results are collected and visualized to assess their effect on model performance. The main implementation is [`oscar.cubeles_lab5.ipynb`](labs/Lab5/oscar.cubeles_lab5.ipynb), accompanied by the [`lab report`](labs/Lab5/oscar.cubeles_lab5.pdf).

### Lab 7 - Crack segmentation with deep learning

Lab 7 implements semantic segmentation of surface cracks in MATLAB using the DeepCrack dataset. It prepares paired image and pixel-label datastores, resizes the data for training, constructs and trains a two-class U-Net, evaluates predictions using global accuracy, intersection over union, and F1 score, and visualizes predicted crack masks against their ground truth. The main implementation is the MATLAB live script [`lab7_deepcrack_oscar_cubeles.mlx`](labs/oscar.cubeles_lab7/lab7_deepcrack_oscar_cubeles.mlx); the folder also contains the trained network, saved predictions, dataset archive, and [`lab report`](labs/oscar.cubeles_lab7/oscar.cubeles_report_lab7.pdf).

### Project 1 - Circle detection in real-world images

Project 1 develops and compares techniques for detecting and counting circular objects under varying image conditions. Starting from the Hough Circle Transform, it explores Canny edges, high-pass filtering, filled edge masks, unsharp masking, adaptive thresholding, morphological preprocessing, contour analysis, and circularity-based detection. The experiments and visual comparisons are in [`assignment.ipynb`](labs/Project1AVPR/assignment.ipynb), with the project brief in [`Task1.pdf`](labs/Project1AVPR/Task1.pdf) and the test images in the [`Images`](labs/Project1AVPR/Images) directory.

## Research Work

The research work reviews **"Recognizing Food Places in Egocentric Photo-Streams Using Multi-Scale Atrous Convolutional Networks and Self-Attention Mechanism"** by Sarker et al., published in *IEEE Access* in 2019 ([DOI: 10.1109/ACCESS.2019.2902225](https://doi.org/10.1109/ACCESS.2019.2902225)). The paper proposes a scene-classification architecture that combines multi-scale atrous convolutions with self-attention to recognize food-related places in lifelogging images. It evaluates the method on the EgoFoodPlaces dataset, which contains 43,392 images captured by 16 participants, and reports 80% overall classification accuracy while outperforming the compared VGG16, ResNet50, and InceptionV3 baselines.

The coursework deliverables are the detailed [`paper summary`](research-work/oscar.cubeles_AVPR_paper_summary.pdf) and the accompanying [`presentation`](research-work/oscar.cubeles_AVPR_presentation.pdf).
