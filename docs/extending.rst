.. _extend:

#########
Extending
#########

You can customize scipr to use your own matching or transformation functions
(e.g. a complicated deep neural network) by subclassing the below *Abstract*
base classes and overriding the necessary methods.

As a rule, the interface is *functional*, and so you are always required to
return the new object state in functions such as :meth:`fit`, instead of
setting object attributes. scipr will take care of managing the object state
for you.

It is recommended that you look at the source code of the provided functions in 
the :ref:`api` and use them as examples of how to implement your own.

*************************
Custom matching functions
*************************

================
Match base class
================

.. autoclass:: scipr.matching.Match
   :members:

*******************************
Custom transformation functions
*******************************

======================
Transformer base class
======================

.. autoclass:: scipr.transform.Transformer
   :members:
