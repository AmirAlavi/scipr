.. _api:

###
API
###

***********
SCIPR Model
***********

.. autoclass:: scipr.SCIPR

   .. rubric:: Methods

   .. autosummary::

      ~fit
      ~fit_adata
      ~transform
      ~transform_adata

   .. automethod:: scipr.SCIPR.fit
   .. automethod:: scipr.SCIPR.fit_adata
   .. automethod:: scipr.SCIPR.transform
   .. automethod:: scipr.SCIPR.transform_adata

******************
Matching Functions
******************

=======
Closest
=======

.. autoclass:: scipr.matching.Closest

======
Greedy
======

.. autoclass:: scipr.matching.Greedy

===
MNN
===

.. autoclass:: scipr.matching.MNN

=========
Hungarian
=========

.. autoclass:: scipr.matching.Hungarian

************************
Transformation Functions
************************

=====
Rigid
=====

.. autoclass:: scipr.transform.Rigid

======
Affine
======

.. autoclass:: scipr.transform.Affine

==================
StackedAutoEncoder
==================

.. autoclass:: scipr.transform.StackedAutoEncoder