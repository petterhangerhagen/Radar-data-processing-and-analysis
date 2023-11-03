# The VIMMJIPDA tracker

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
* `plotting.py`: contains the plotting functionality.
