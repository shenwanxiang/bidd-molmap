https://tendcode.com/article/setup-to-pypy/

- 1. config the setup.py, MANIFEST.in
- 2. python setup.py sdist bdist_wheel
- 3. python setup.py install # test, installed
- 4. pip uninstall aggmap #uninstall
- 5. twine upload dist/* #upload to pypi 

