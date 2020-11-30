Installation
============

Installing molmap is pretty simple. Here is a step by step plan on how to do it.

.. note::
    Kong is available on Pypi as ``django-kong``, but trunk is probably your
    best best for the most up to date features.

First, obtain Python_ and virtualenv_ if you do not already have them. Using a
virtual environment will make the installation easier, and will help to avoid
clutter in your system-wide libraries. You will also need Git_ in order to
clone the repository.

.. _Python: http://www.python.org/
.. _virtualenv: http://pypi.python.org/pypi/virtualenv
.. _Git: http://git-scm.com/

Once you have these, create a virtual environment somewhere on your disk, then
activate it::

    virtualenv kong
    cd kong
    source bin/activate


Kong ships with an example project that should get you up and running quickly. To actually get kong running, do the following::

    git clone http://github.com/ericholscher/django-kong.git
    cd django-kong
    pip install -r requirements.txt
    pip install . #Install Kong
    cd example_project
    ./manage.py syncdb --noinput
    ./manage.py loaddata test_data
    ./manage.py runserver


This will give you a locally running instance with a couple of example sites
and an example test.

Now that you have your tests in your database, you need to check that your
tests run. You can run tests like::

    #Check all sites
    ./manage.py check_sites
    #Only run the front page test
    ./manage.py check_sites -t front-page
    #Only check sites of type Mine
    ./manage.py check_sites -T mine

The first command is the default way of running kong, and will run the tests for all of your sites.

The second two different ways will run either a specific test, or a type of test. Both of these can run tests across multiple sites.
    
