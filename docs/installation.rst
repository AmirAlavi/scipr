.. highlight:: shell

.. _install:

============
Installation
============


Stable release
--------------

To install scipr, run this command in your terminal:

.. code-block:: console

    $ pip install scipr

If you would like to use the optional TensorBoard logging features of scipr,
install scipr with this command instead:

.. code-block:: console

    $ pip install scipr[tensorboard]

The above are the preferred methods to install scipr, as they will always
install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for scipr can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/amiralavi/scipr

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/amiralavi/scipr/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/amiralavi/scipr
.. _tarball: https://github.com/amiralavi/scipr/tarball/master
