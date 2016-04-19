import os
from ctypes import *

print(os.getcwd())
print(os.listdir())
# load the shared object file
parda = CDLL('./pardaFolder.so')

l = c_long(137979)
f = c_char_p("normal_137979.trace")
res_int = parda.classical_tree_based_stackdist(4, l)
