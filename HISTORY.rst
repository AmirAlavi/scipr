=======
History
=======

0.1.0 (2020-05-13)
------------------

* First release on PyPI.

0.2.0 (2020-05-22)
------------------

* New logging capabilities
    - scipr API calls will no longer result in any output to stdout under normal conditions
    - Instead, now using Python's built-in `logging` module to emit logging information
    - Can also track SCIPR model fitting metrics per iteration in TensorBoard (optional)
* Now compatible with AnnData objects for fitting and transforming

0.2.1 (2020-05-26)
------------------

* Fix for bug that would cause MNN to break if `k=1`
* Fix for mistake in SCIPR.transform_adata documentation about Returns
* More tests for matching algorithms added
