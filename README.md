# GVS: Gait Video Study 

### Authors:
* Rachneet Kaur rk4@illinois.edu https://www.linkedin.com/in/rachneet-kaur-a1ba5354/
* Manuel Hernandez mhernand@illinois.edu, http://kch.illinois.edu/hernandez
* Richard Sowers r-sowers@illinois.edu, http://publish.illinois.edu/r-sowers/

### Code structure:
#### Data Preperation 
* **Downsample.ipynb**: To convert videos to frames at a 30FPS (frames per second) and downsample if needed. 
* **OpenPose2Dkeypoints.ipynb**: To compute 2D coordinates from extracted frames using OpenPose package https://github.com/CMU-Perceptual-Computing-Lab/openpose.
* **CameraCalibartion.ipynb**: To compute the intrinsic and extrinsic camera matrices for converting the 2D pixel units to centimeters (cm) in 3D world coordinates. We consider the origin (x=0, y=0, z=0 set at the start of the treadmill) of the treadmill as the origin of the world coordinates.
* **PostProcessing2Ddata.ipynb**: To manage left-right swap issue and missing value treatment for lower body frames and retaining the coorect body markers for the feet frames. Also computes the statistics of the postprocessing steps. 
* **2Dto3D_subject_wise_checks.ipynb**: To double check the extrinsic camera parameters (computed rotation and translation matrices) for both the front and side camera for each subject. This check is handy when say a couple of subjects had a slightly different orientation of the camera than it is calibrated for. In such cases then, we can adjust the parameters for these subjects to avoid incorrect 3D coordinates. 
* **Convert2Dto3D.ipynb**: To use the computed intrinsic and extrinsic camera matrices (via the CameraCalibration.ipynb) for both the front and side camera to convert all the extracted 2D pose coordinates to 3D real-world pose coordinates http://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf.
* **Postprocessing3Ddata.ipynb**: Postprocessing the created 3D keypoints (via intrinsic and extrinsic matrix) for lower body and feet and combining together both front and side view 3D coordinates to get the final combined coordinates. Further, scaling all average hip heights to a constant to normalize for subject heights in our dataset.
* **IdentifyHSR.ipynb**: Identifying frames with HSRs in each video for each cohort and trial to establish break points and also evaluate the corresponding HSR labelling via the ground truth available. Further, downsampling with smoothing to define fixed shape of the input tensor for models. 
* **StrideStats.ipynb**: Calculating the stats for strides in each framework (HOA-BW/W, MS-BW/W, PD-BW/W). This will help write stats for count of strides used in training and testing set of each framework, 1. task generalization a) W to WT, and b) T to TT and 2. subject generalization a) W, b) WT, c) T, and d) TT.

#### CoP validation
To validate the estimates 3D poses via CoP computed on the treadmill 
* **CoPvalidation_make_viz_dataframes.ipynb**: This code makes (and saves as csv) the dataframe for each video containing their relevant frames, corresponding feet coordinates for these frames and their relative treadmill extracted CoP values 
* **CoPvalidation_evaluate_visually_quantitatively.ipynb**: This code qualitatively and quantitatively validates the center of mass trajectory of estimates body coordinates with their respective treadmill extracted CoP's on a frame-wise/support group-wise basis. This code creates the csv files to further do the final qualitative and quantitative analysis in CoPvalidation_result_analysis.ipynb.
* **CoPvalidation_result_analysis.ipynb**: This code will do the final qualitative and quantitative validation for CoP and centroid of BoS. We compile the quantitative results of distances and correlations between CoPs and centroids for each cohort, trial (W/WT) and support type (Single/Double). Further, we will plot the butterfly diagrams for CoPs and centroids to qualitatively vizualize how treadmill CoPs match with BoS's centroid values. 

#### Machine Learning 
* **imports.py**: package imports
* **split.py**: Function definition for StratifiedGroupKFold for handling imbalance in case of Group K fold https://github.com/scikit-learn/scikit-learn/issues/13621
* **SummaryStats.ipynb**: Creating the summary statistics file for the traditional ML algorithms on task/subject generalization frameworks. We use the summary statistics as range, CoV and asymmetry between the right and left limbs as the features to input to the traditional models requiring fixed size 1D input for each training/testing set sample.
* **TaskGeneralize_MLtraditional.ipynb**: Traditional ML algorithms on task generalization framework 1: train on walking (W) and test on walking while talking (WT) and 2: train on virtual beam walking (VBW) and test on virtual beam walking while talking (VBWT) to classify HOA/MS/PD strides and subjects. We use majority voting for subject classification. We retain only subjects common to both train and test sets for this analysis. 
* **SubjectGeneralize_MLtraditional.ipynb**: Traditional ML algorithms on subject generalization frameworks, 1: W, 2: WT, 3: VBW, 4: VBWT using cross validation (we use stratified group K folds here) to classify HOA/MS/PD strides and subjects. We use majority voting for subject classification. Further, to compare across the four sub-frameworks of subject generalization, we retain only common subjects across the four frameworks and then rank the frameworks on the basis of best to worst subject generalization performance/capability.
* **Task_and_SubjectGeneralize_MLtraditional.ipynb**: Traditional ML algorithms on task+subject generalization frameworks, 1. train on some subjects in W and test on separate set of subjects in WT, 2. train on some subjects in VBW and test on separate set of subjects in VBWT, to classify HOA/MS/PD strides and subjects. We use majority voting for subject classification. We use cross validation here but further retain only subjects present for W in training for 1. and only subjects present for WT in testing for 1., and similarly for 2.
* **TaskGeneralize_Conv1D.ipynb**: 1D CNN algorithm on task generalization framework 1: train on walking (W) and test on walking while talking (WT) and 2: train on virtual beam walking (VBW) and test on virtual beam walking while talking (VBWT) to classify HOA/MS/PD strides and subjects. 
* **SubjectGeneralize_Conv1D.ipynb**: 1D CNN algorithm on subject generalization frameworks, 1: W, 2: WT, 3: VBW, 4: VBWT using cross validation (we use stratified group K folds here) to classify HOA/MS/PD strides and subjects.

#### Discussion analysis
*  **FeatureImportance.ipynb**: 

### Citation:
If you use this code, please consider citing our work:
```

```
