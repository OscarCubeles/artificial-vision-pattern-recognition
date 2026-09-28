# Artificial Vision and Pattern Recognition

This repository contains the coursework completed for the **Artificial Vision and Pattern Recognition** subject. It brings together the practical laboratory exercises, a larger image-processing project, and a research study based on a published computer-vision paper. 

## Labs

1. **Lab 1 - Image filtering:** Compares edge detectors, custom kernels, denoising filters, motion blur, and frequency-domain filtering. See the [`source notebook`](labs/Lab1/lab1.ipynb) and [`exercise document`](labs/Lab1/LAB_1.pdf).

2. **Lab 2 - Edges, corners, and morphology:** Explores Canny edge detection, Harris corners, and morphological operations such as erosion, dilation, opening, and closing. See the [`source notebook`](labs/Lab2/lab2.ipynb), [`Python export`](labs/Lab2/lab2.py), and [`report`](labs/Lab2/oscar.cubeles_lab2_AVPR.pdf).

3. **Lab 3 - Feature extraction and matching:** Uses SIFT and LBP descriptors for feature matching, image alignment, and texture classification. See the [`source notebook`](labs/Lab3/Lab3AVPR_oscar.cubeles.ipynb) and [`report`](labs/Lab3/Lab3AVPR_oscar.cubeles.pdf).

4. **Lab 5 - ResNet experimentation:** Studies how hyperparameters, architectural changes, and data augmentation affect a PyTorch ResNet image classifier. See the [`source notebook`](labs/Lab5/oscar.cubeles_lab5.ipynb) and [`report`](labs/Lab5/oscar.cubeles_lab5.pdf).

5. **Lab 7 - Crack segmentation:** Trains and evaluates a MATLAB U-Net for semantic segmentation of surface cracks from the DeepCrack dataset. See the [`MATLAB live script`](labs/oscar.cubeles_lab7/lab7_deepcrack_oscar_cubeles.mlx) and [`report`](labs/oscar.cubeles_lab7/oscar.cubeles_report_lab7.pdf).

6. **Project 1 - Circle detection:** Compares Hough transforms, edge-based preprocessing, adaptive thresholding, and contour analysis for detecting circular objects. See the [`source notebook`](labs/Project1AVPR/assignment.ipynb), [`project brief`](labs/Project1AVPR/Task1.pdf), and [`test images`](labs/Project1AVPR/Images).

## Research Work

The research work reviews **"Recognizing Food Places in Egocentric Photo-Streams Using Multi-Scale Atrous Convolutional Networks and Self-Attention Mechanism"** by Sarker et al., published in *IEEE Access* in 2019 ([DOI: 10.1109/ACCESS.2019.2902225](https://doi.org/10.1109/ACCESS.2019.2902225)). The paper proposes a scene-classification architecture that combines multi-scale atrous convolutions with self-attention to recognize food-related places in lifelogging images. It evaluates the method on the EgoFoodPlaces dataset, which contains 43,392 images captured by 16 participants, and reports 80% overall classification accuracy while outperforming the compared VGG16, ResNet50, and InceptionV3 baselines.

The coursework deliverables are the detailed [`paper summary`](research-work/oscar.cubeles_AVPR_paper_summary.pdf) and the accompanying [`presentation`](research-work/oscar.cubeles_AVPR_presentation.pdf).
