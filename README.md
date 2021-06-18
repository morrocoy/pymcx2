# pymcx2

Pymcx2 is a cross-platform python interface package for [Monte Carlo eXtreme (MCX) - 
CUDA Edition](https://github.com/fangq/mcx).

TODO

## Installation


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

