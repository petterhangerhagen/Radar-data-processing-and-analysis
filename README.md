# Processing and analysis of radar dataset
This code is processing and analyzing a radar dataset. The dataset is recorded from the Trondheim City Canal during the summer of 2023. The dataset can be obtain from ..., and showed have the following file path ...

Large parts of the code is adapted from the Code Ocean capsule by Audun Gullikstad Hem (2021), https://codeocean.com/capsule/3448343/tree/v1.

## How to use the code:
I am using a virtual python environment to get correct version of the requierments, without messig up for other projects. A recommondation, but not nescessary. 

The required packages can be found inside `requirements.txt`. All can be installed by running:
```pip install -r /path/to/requirements.txt ```

Inside `run.py` its important to change the `radar_data_path` and the `working_directory` to match the structure you have!

## Structure
The code is launched from `run.py`. Here, one can choose which scenarios form the dataset to be used and include different different analysis. The code inside *tracking*-folder is a pure copy from Audun Gullikstad Hem, and follows the same structure:

* `associators.py`: contains data associators.
* `constructs.py`: contains the different data structures the tracker uses.
* `filters.py`: contains the state filters.
* `initiators.py`: contains functionality for initiating tracks.
* `managers.py`: contains the manager, which calls the initiators, terminators and trackers.
* `models.py`: contains the different models, i.e. kinematic models, measurement models and clutter models.
* `terminators.py`: contains functionality for terminating tracks.
* `trackers.py`: contains the tracker itself.
* `utilities.py`: contains helping functions that are being utilized by the other modules.

The supporting structure from Audun Gullikstad Hem have been edited to fit the new dataset. 

* `import_radar_data.py`: Imports the radar data from .json files and converts into desired format. 
* `parameters.py`: contains all the paramets for the tracker. The parameters can be changed in this file.
* `plotting.py`: contains the plotting functionality.
* `video.py`: contains functionality for creating video of tracking scenarios. 

New custom defined functionality is contained in the *utilities*-folder:
* `utilities.py`: contains support functions and different functions used for analysing the data.
* `images_to_video.py`: contains support functions for creating a video.
* `check_start_and_stop.py`: contains functionality for creating traffic matrix based on tracks established by the tracker.
* `merged_measurements/`: 
    * `merged_measurements.py`: containg functionality for finding merged measurements.
    * `merged_measurements.txt`: contains the name of the .json files which are defiend as merged measurements scenarios.
    * `merged_measurement/`: 
* `multi_path/`: 
    * `multi_path.py`: contains functionality for findig multi path scenarios.
    * `multi_path.txt`: contains the name of the .json files which are defiend as multi path scenarios.
    * `multi_path/`: 
* `multi_target/`:
    * `multi_target_scenarios.py`: contains functionality for findig multi-target scenarios.
    * `multi_target_scenarios.txt`: contains the name of the .json files which are defiend as multi-target scenarios.
    * `multi_target/`: 


The *npy_files*-folder contains .npy files used for storing data temporary. 







<!-- # The VIMMJIPDA tracker

This code is all that is required to run the VIMMJIPDA tracker described in "Multi-target tracking with multiple models and
visibility verified on maritime radar data". Two data sets are included, which are both described in the aforementioned article.

## How to use the code:

This code requires numpy, matplotlib, scipy, anytree and Shapely. All can be installed by running
`pip install -r /path/to/requirements.txt`. 

The algorithm is launched by running `run.py`. Here, one can choose which data set to use, and whether any of the properties of the tracker should be removed. One can, for example, to remove the IMM-, multi-target- and visibility-functionality to reduce the tracker to an IPDA. Furthermore, the code is designed to be modular. As such, it should be possible to implement new clutter models, measurement models etc. without too much problems.

## Structure:

The code for the tracker itself is contained in the *tracking*-folder. It contains nine files:
* `associators.py`: contains data associators.
* `constructs.py`: contains the different data structures the tracker uses.
* `filters.py`: contains the state filters.
* `initiators.py`: contains functionality for initiating tracks.
* `managers.py`: contains the manager, which calls the initiators, terminators and trackers.
* `models.py`: contains the different models, i.e. kinematic models, measurement models and clutter models.
* `terminators.py`: contains functionality for terminating tracks.
* `trackers.py`: contains the tracker itself.
* `utilities.py`: contains helping functions that are being utilized by the other modules.

Furthermore, there is some supporting structure:

* `import_data.py`: imports the data from the .mat files in the data folder and converts it to the desired form.
* `parameters.py`: contains all the paramets for the tracker. The parameters can be changed in this file, and then imported to various run-scripts.
* `plotting.py`: contains the plotting functionality. -->

