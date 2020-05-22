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
