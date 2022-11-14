import numpy as np
from sympy import Matrix
import sympy
import pprint


#A = np.array([[3,1,0,0],[-4,-1,0,0],[6,2,0,-1],[-2,0,1,2]])
A = np.array([[-2, 1, 0], [-2, 1, -1], [-1, 1, -2]])
#A = np.array([[-2, 1, 0], [-2, 1, -1], [-1, 1, -2]])

a = Matrix(A)
P, Ja = a.jordan_form()

pprint.pprint(Ja)
pprint.pprint(P)