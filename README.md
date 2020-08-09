# GVS: Gait Video Study 

### Authors:
* Rachneet Kaur rk4@illinois.edu https://www.linkedin.com/in/rachneet-kaur-a1ba5354/
* Zizhang Chen zizhang2@illinois.edu
* Manuel Hernandez mhernand@illinois.edu, http://kch.illinois.edu/hernandez
* Richard Sowers r-sowers@illinois.edu, http://publish.illinois.edu/r-sowers/

### Data Preperation 
* **Downsample.ipynb**: To convert videos to frames at a 30FPS (frames per second) and downsample if needed. 
* **OpenPose2Dkeypoints.ipynb**: To compute 2D coordinates from extracted frames using OpenPose package https://github.com/CMU-Perceptual-Computing-Lab/openpose.
* **CameraCalibartion.ipynb**: To convert pixel units to centimeters (cm) in world coordinates considering the origin of the treadmill as the origin of the world coordinates.
* **PostProcessing2Ddata.ipynb**: To manage left-right swap issue and missing value treatment for lower body frames and retaining the coorect body markers for the feet frames. Also computes the statistics of the postprocessing steps. 

### Machine Learning 

### Discussion analysis

