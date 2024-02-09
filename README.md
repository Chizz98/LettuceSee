# LettuceSee
A package with high level functions for analyzing images of plants.

## Example case
The following example showcases a miniature pipeline using lettuceSee and 
scikit-image functions. 

![Alt text](https://github.com/Chizz98/LettuceSee/blob/main/readme_images/original_image.png "Input image")

An image of lettuce affected with necrosis is loaded in as a numpy array. 
After removing the alpha layer, applying the shw_segmentation function from the
segment module removes the background of the image. 

```python
from lettuceSee import segment
import skimage
import matplotlib.pyplot as plt

image = skimage.io.imread(
    r"C:\Users\chris\Documents\GitHub\tipburn_quantification\test_images"
    r"\rgb\51-78-Lettuce_Correct_Tray_074-RGB-Original_pos2_LK120.png")
image = skimage.util.img_as_ubyte(skimage.color.rgba2rgb(image))
bg_mask = segment.shw_segmentation(image)
```

![Alt text](/readme_images/bg_mask1.png "Background mask")

The initial segmentation looks decent, however there is still some background 
noise, as well as other plants intruding from the side. The function 
canny_central_ob is used to remove objects not attached to the central object, 
as well as connected objects with very different coloration.
```python
bg_mask = segment.canny_central_ob(image=image, mask=bg_mask, sigma=2.5)
```

![Alt text](/readme_images/bg_mask2.png "Cleaned background mask")

After finishing the background segmentation, the function barb_hue is used to 
segment green from brown tissue through the method described in 
[Barbedo, 2016](https://doi.org/10.1007/s40858-016-0090-8).

```python
necrosis_mask = segment.barb_hue(image=image, bg_mask=bg_mask, div=3)
```
![Alt text](/readme_images/necrosis_mask.png "Background + necrosis mask")

## Installation
The package can be installed from the pypi test distribution trough:
```shell
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ lettuceSee=0.0.12
```

### Anaconda
There is no dedicated lettuceSee installation for anaconda, if you do want to 
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

### Recommended extras
For visualization of the images, matplotlib is recommended. LettuceSee handles 
images as numpy arrays, which can be directly visualized trough 
matplotlib.pyplot. Matplotlib is not included in the installation of lettuceSee,
but can be installed trough:
```shell
pip install matplotlib
```