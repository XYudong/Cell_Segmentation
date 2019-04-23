# Segmentation of Speroid Cancer Cells


## Getting started
[data_utils.py](data_utils.py): for most data processing works;

[Unet_training.py](Unet_training.py): builds up the model and trains it;

[Unet_evaluation.py](Unet_evaluation.py): runs tests and visualize the evaluation results in different forms;

[cell_analysis.py](cell_analysis.py): for analysis of the segmented masks;

[analysis_utils.py](analysis_utils.py): implements extracting various morphological features

## Results

Samples from Test set:
* Overlay predicted masks with raw images:
![C3](pics/C3_rawAndMask.png)
![C10](pics/C10_rawAndMask.png)
