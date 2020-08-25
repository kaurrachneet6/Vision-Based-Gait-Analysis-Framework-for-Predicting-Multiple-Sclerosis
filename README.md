# GVS: Gait Video Study 

### Authors:
* Rachneet Kaur rk4@illinois.edu https://www.linkedin.com/in/rachneet-kaur-a1ba5354/
* Zizhang Chen zizhang2@illinois.edu
* Manuel Hernandez mhernand@illinois.edu, http://kch.illinois.edu/hernandez
* Richard Sowers r-sowers@illinois.edu, http://publish.illinois.edu/r-sowers/

### Data Preperation 
* **Downsample.ipynb**: To convert videos to frames at a 30FPS (frames per second) and downsample if needed. 
* **OpenPose2Dkeypoints.ipynb**: To compute 2D coordinates from extracted frames using OpenPose package https://github.com/CMU-Perceptual-Computing-Lab/openpose.
* **CameraCalibartion.ipynb**: To compute the intrinsic and extrinsic camera matrices for converting the 2D pixel units to centimeters (cm) in 3D world coordinates. We consider the origin (x=0, y=0, z=0 set at the start of the treadmill) of the treadmill as the origin of the world coordinates.
* **PostProcessing2Ddata.ipynb**: To manage left-right swap issue and missing value treatment for lower body frames and retaining the coorect body markers for the feet frames. Also computes the statistics of the postprocessing steps. 
* **2Dto3D_subject_wise_checks.ipynb**: To double check the extrinsic camera parameters (computed rotation and translation matrices) for both the front and side camera for each subject. This check is handy when say a couple of subjects had a slightly different orientation of the camera than it is calibrated for. In such cases then, we can adjust the parameters for these subjects to avoid incorrect 3D coordinates. 
* **Convert2Dto3D.ipynb**: To use the computed intrinsic and extrinsic camera matrices (via the CameraCalibration.ipynb) for both the front and side camera to convert all the extracted 2D pose coordinates to 3D real-world pose coordinates http://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf.
* **Postprocessing3Ddata.ipynb**: Postprocessing the created 3D keypoints (via intrinsic and extrinsic matrix) for lower body and feet and combining together both front and side view 3D coordinates to get the final combined coordinates. Further, scaling all average hip heights to a constant to normalize for subject heights in our dataset.
* **IdentifyHSR.ipynb**: Identifying frames with HSRs in each video for each cohort and trial to establish break points and also evaluate the corresponding HSR labelling via the ground truth available. Further, downsampling with smoothing to define fixed shape of the input tensor for models. 
* **StrideStats.ipynb**: Calculating the stats for strides in each framework (HOA-BW/W, MS-BW/W, PD-BW/W). This will help write stats for count of strides used in training and testing set of each framework, 1. task generalization a) W to WT, and b) T to TT and 2. subject generalization a) W, b) WT, c) T, and d) TT.

### Machine Learning 
* **TaskGeneralizeWtoWT_MLtraditional.ipynb**:
* **SubjectGeneralizeW_MLtraditional.ipynb**:

### Discussion analysis

