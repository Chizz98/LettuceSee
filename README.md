# LettuceSee
A package developed for the analysis of plant images.

## Installation
The package can be installed from the pypi test distribution trough:
```shell
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ lettuceSee=0.0.11
```

### Anaconda
There is no dedicated lettucesee installation for anaconda, if you do want to 
install the package within anaconda the following method is recommended:
First create a new environment following the [anaconda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands). 
Activate your fresh environment and install pip:
```shell
conda install pip
```
Following this run the usual command to install lettuceSee. Note that mixing 
conda and pip can lead to unexpected errors, as both are package managers. As 
such it is recommended to do further installations in this environment trough 
pip.

### Reccomended extras
For visualization of the images, matplotlib is recommended. LettuceSee handles 
images as numpy arrays, which can be directly visualized trough 
matplotlib.pyplot. Matplotlib is not included in the installation of lettuceSee,
but can be installed trough:
```shell
pip install matplotlib
```