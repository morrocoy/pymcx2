.. _api_misc_find_mcx:

Find MCX
========

When importing the package pymcx2 it searches in `sys.path` for the mxc binary.
Internally the function `find_mcx()` is called which returns on success the path
to the binary. Thus, make sure that either the root or bin
path of MCX is within `sys.path` before importing the package.
Otherwise the following warning will pop up during the import:

.. code-block:: python

    >>> import pymcx2
    Warning: Could not find path to mcx binary.


.. autofunction:: pymcx2.find_mcx
