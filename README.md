# ml_tools
collection of tools to predict atomic properties

Requirements:

+ python 3.*

+ quippy (optional) (see http://libatoms.github.io/QUIP/install.html for installation and prefere the openmp arch builds)

+ numpy scipy scikit-learn autograd pandas tqdm ase numba, e.g.
```
pip install ase
conda install -c conda-forge numpy scipy scikit-learn autograd pandas tqdm numba
```

To use the SOAP as implemented by Dr. Michael Willat run:
```
make
```