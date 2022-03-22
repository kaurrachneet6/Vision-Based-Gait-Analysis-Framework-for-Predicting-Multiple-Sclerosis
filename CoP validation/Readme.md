
#### CoP validation
To validate the estimates 3D poses via CoP computed on the treadmill 
* **CoPvalidation_make_viz_dataframes.ipynb**: This code makes (and saves as csv) the dataframe for each video containing their relevant frames, corresponding feet coordinates for these frames and their relative treadmill extracted CoP values 
* **CoPvalidation_evaluate_visually_quantitatively.ipynb**: This code qualitatively and quantitatively validates the center of mass trajectory of estimates body coordinates with their respective treadmill extracted CoP's on a frame-wise/support group-wise basis. This code creates the csv files to further do the final qualitative and quantitative analysis in CoPvalidation_result_analysis.ipynb.
* **CoPvalidation_result_analysis.ipynb**: This code will do the final qualitative and quantitative validation for CoP and centroid of BoS. We compile the quantitative results of distances and correlations between CoPs and centroids for each cohort, trial (W/WT) and support type (Single/Double). Further, we will plot the butterfly diagrams for CoPs and centroids to qualitatively vizualize how treadmill CoPs match with BoS's centroid values. 
