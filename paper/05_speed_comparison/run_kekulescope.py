import os, sys
import molmap
from molmap import dataset
path = os.path.join(os.path.dirname(os.path.dirname(molmap.__file__)), 
                    'paper/03_KekuleScope_comparison/kekulescope')
sys.path.insert(0, path)