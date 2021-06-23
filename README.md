# pymcx2

PyMCX2 is a cross-platform python interface package for [Monte Carlo eXtreme (MCX) - 
CUDA Edition](https://github.com/fangq/mcx).

TODO


## Requirements

This project supports the following versions of python:
* Python 2.7+ and Python 3.7+
* Required packages:
  * `numpy`
* Optional packages:
  * `pandas` for detected photons
    
It has been tested on Ubuntu 20.04 and Windows 10.
     
## Installation

TODO 

## Link to MCX
When importing the package pymcx2 it searches in `sys.path` for the mxc binary. 
Internally the function `find_mcx()` is called which returns on success the path 
to the binary. Thus, make sure that either the root or bin
path of MCX is within `sys.path` before importing the package.
Otherwise the following warning will pop up during the import: 
```
>>> import pymcx2
Warning: Could not find path to mcx binary.
```

## Running Simulations
TODO

## Detected Photons

The simulation results of all detected photons are available from a pandas 
dataframe via the member `detected_photons` of the session:
```
dp = session.detected_photons
```  
Apart from the index, the dataframe generally provides the following fields with 
the corresponding number of columns in brackets: 
* `detectid` - The detector ID (1).
* `nscatter_mat{1, ..., N}` - The partial scattering event counts (No. of materials `N`).
* `ppathlen_mat{1, ..., N}` - The partial path-lengths (No. of materials `N`).
* `momentum_mat{1, ..., N}` - The momentum transfer (No. of materials `N`).
* `pos_exit_{x, y, z}` - The exit position (3).
* `dir_exit_{x, y, z}` - The exit direction (3).
* `weight` - The initial photon weight (1).

Please refer to the [MCX documentation](http://mcx.space/wiki/index.cgi?Doc/mcx_help#savedetflag) 
for further details on the different fields.
A simulation involving two different materials and a detector may provide a 
dataframe according to the following structure: 

| Index | detectid  | nscatter_mat1 | nscatter_mat2 | ppathlen_mat1 | ppathlen_mat2 | momentum_mat1 | momentum_mat2 | pos_exit_x | pos_exit_y | pos_exit_z | dir_exit_x | dir_exit_y | dir_exit_z | weight |
| :---: | :-------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :----: |
| 0 | 1.0 | 6.0 | 0.0 | 30.526659 | 0.0 | 1.4141142 | 0.0 | 94.63195 | 72.35736 | 0.99993896 | -0.39827162 | -0.6059654 | -0.6886109 | 1.0 |
| 1 | 1.0 | 3.0 | 0.0 | 28.041206 | 0.0 | 0.5588621 | 0.0 | 79.318085 | 87.4064 | 11.000061 | -0.8471075 | -0.4603843 | 0.2654321 | 1.0 |
| 2 | 1.0 | 4.0 | 0.0 | 27.098019 | 0.0 | 1.2287982 | 0.0 | 85.86776 | 82.88043 | 11.000061 | -0.27062535 | -0.42192587 | 0.86529773 | 1.0 |
| 3 | 1.0 | 4.0 | 0.0 | 28.093813 | 0.0 | 1.0805666 | 0.0 | 84.57262 | 83.26662 | 11.000061 | -0.4139344 | -0.5780247 | 0.70323914 | 1.0 |
 
Note, if no detector is defined MCX will not create any `*.mch` file and 
`session.detected_photons` becomes `None`.

Common postprocessing analyses may efficiently be applied on
the dataframe directly. The following code snipped allows to recalculate the 
detected photon weight using partial path data and optical properties: 
```
dp = session.detected_photons
scale = session.domain["scale"]
for mat in session.material.values():
    if mat["tag"] != 0:
        dp["weight"] = dp.apply(lambda dp: dp["weight"] * np.exp(
            -mat["mua"] * scale * dp["ppathlen_mat%d" % (mat["tag"])]), axis=1)
```
It corresponds to the matlab function in `mcxdetweight.m` provided by MCX. It 
is important to note that the background material with the tag 0 must be
excluded here.

## Links
- [Monte Carlo eXtreme (MCX) - CUDA Edition](https://github.com/fangq/mcx).
- [Command line options for MCX](http://mcx.space/wiki/index.cgi?Doc/mcx_help#Command_Line_Options).
- [PyMCX - a function based python interface package for MCX](https://github.com/4D42/pymcx).
