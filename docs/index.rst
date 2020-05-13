#################
Welcome to scipr!
#################

.. image:: https://img.shields.io/pypi/v/scipr.svg
        :target: https://pypi.python.org/pypi/scipr

.. image:: https://img.shields.io/travis/amiralavi/scipr.svg
        :target: https://travis-ci.com/amiralavi/scipr

.. image:: https://readthedocs.org/projects/scipr/badge/?version=latest
        :target: https://scipr.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. toctree::
   :maxdepth: 3
   :hidden:

   self
   intro
   installation
   api
   extending
   contributing
   authors
   history

**************
What is scipr?
**************

Single Cell Iterative Point set Registration (scipr) learns a transformation
function to align batches of scRNA-seq data. This site covers scipr's usage &
API documentation.

***********
Quick start
***********

Learn a function to align to batches of scRNA-seq data::

   import scipr
   from scipr.matching import MNN
   from scipr.transform import Affine

   model = scipr.SCIPR(match_algo=MNN(), transform_algo=Affine())
   model.fit(batch_A, batch_B)

Apply the function to align data::

   batch_A_aligned = model.transform(batch_A)

**********
Learn more
**********

Read more about how scipr works in the :ref:`get-started` guide. For
documentation on each of the functions available in scipr, see the :ref:`api`.
This package is based on the methods presented in our bioRxiv paper
(forthcoming).

scipr is made available for free under a BSD license.

****************
Related projects
****************

ScRNA-seq alignment (or "batch correction") methods is an active area of
developement. Here are just a few other related works you might be interested
in:

* `Seurat v3 <https://satijalab.org/seurat/>`_ 
* `scAlign <https://github.com/quon-titative-biology/scAlign>`_
* `Harmony <https://portals.broadinstitute.org/harmony/>`_
* `mnnpy <https://github.com/chriscainx/mnnpy>`_

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
