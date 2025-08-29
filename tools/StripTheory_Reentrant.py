from __future__ import division
from scipy import sparse,spatial
import numpy as np
import time
from scipy import interpolate
import sympy as sym
import numpy as np
from math import *
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from scipy.integrate import quad

b=200#
t=6
# S= np.zeros([6,6]) # Original 6x6 S matrix initialization

def Isotropic(E,nu):
    Ed=E
    S=np.array([[1.0/Ed,-nu/Ed,-nu/Ed,0.,0.,0.],
                [-nu/Ed,1.0/Ed,-nu/Ed,0.,0.,0.],
                [-nu/Ed,-nu/Ed,1.0/Ed,0.,0.,0.],
                [0.,0.,0.,(2*(1.0+nu)/Ed),0.,0.],
                [0.,0.,0.,0.,(2*(1.0+nu)/Ed),0.],
                [0.,0.,0.,0.,0.,(2*(1.0+nu)/Ed)]])
    D=np.linalg.inv(S)
    return D
Q=np.zeros([3,3])
C=Isotropic(7e+10,0.37)
Q[0,0]=C[0,0]-(C[0,2]*C[0,2])/C[2,2]
Q[0,1]=C[0,1]-(C[0,2]*C[1,2])/C[2,2]
Q[1,0]=Q[0,1]
Q[1,1]=C[1,1]-(C[1,2]*C[1,2])/C[2,2]
Q[2,2]=C[5,5]
print(Q)
#A(2,2)
A11=Q[0,0]*t
A12=Q[0,1]*t
A16=Q[0,2]*t
A22=Q[1,1]*t
A26=Q[1,2]*t
A66=Q[2,2]*t
D11=Q[0,0]*t*t*t/12
D12=Q[0,1]*t*t*t/12
D16=Q[0,2]*t*t*t/12
D22=Q[1,1]*t*t*t/12
D26=Q[1,2]*t*t*t/12
D66=Q[2,2]*t*t*t/12
#print(D66)
B11=0
B12=0
B16=0
B22=0
B26=0
B66=0
#STFFNESS BAR ELEMENTS-----------------------------------------------------------------

barA11 = A11 + (pow(A16,2)*A22 - 2*A12*A16*A26 + pow(A12,2)*A66)/(pow(A26,2) - A22*A66)
barB11= B11 + (-A16*A26*B12 + A12*A66*B12 + A16*A22*B16 - A12*A26*B16)/(pow(A26,2) - A22*A66)
barB12=B12 + (-A16*A26*B22 + A12*A66*B22 + A16*A22*B26 - A12*A26*B26)/(pow(A26,2) - A22*A66)
barD11=(A66*pow(B12,2) - 2*A26*B12*B16 +A22*pow(B16,2))/(pow(A26,2) - A22*A66) + D11
barD12=(A66*B12*B22 +A22*B16*B26 - A26*(B16*B22 + B12*B26))/(pow(A26,2) - A22*A66) + D12
barD22=(A66*pow(B22,2) - 2*A26*B22*B26 + A22*pow(B26,2))/(pow(A26,2) - A22*A66) + D22
barB16=B16 + ((-A16)*A26*B26 + A12*A66*B26 + A16*A22*B66- A12*A26*B66)/(pow(A26,2) - A22*A66)
barD16=(A66*B12*B26 + A22*B16*B66 - A26*((B16*B26 + B12*B66)))/(pow(A26,2) - A22*A66) + D16
barD26=(A66*B22*B26 + A22*B26*B66- A26*((pow(B26,2)+ B22*B66)))/(pow(A26,2) - A22*A66) + D26
barD66=-(A26*B26*B66)/(pow(A26,2) - A22*A66) + D66
bar2A11=barA11-pow(barB12,2)/barD22
bar2B11=barB11-(barB12*barD12)/barD22
bar2B16=barB16-(barB12*barD26)/barD22
bar2D11=barD11-pow(barD12,2)/barD22
bar2D16=barD16-(barD12*barD26)/barD22
bar2D66=barD66-pow(barD26,2)/barD22

def Crossection_9x9(Ki,b):
    S = np.zeros([9,9]) # Initialize a 9x9 matrix

    # Calculate all Snmn terms as per the original script
    S[0,0] = b*bar2A11
    S[0,1] = - 2*b*bar2B16 +pow(b,3)*Ki*(bar2A11)/12
    S[0,2] = b*bar2B11
    S[0,4] = pow(b,3)*bar2A11/24

    S[1,1]= 4*b*bar2D66-pow(b,3)*bar2B16*Ki/3+pow(b,5)*bar2A11*pow(Ki,2)/80
    S[1,2] = -2*b*bar2D16+pow(b,3)*bar2B11*Ki/12
    S[1,4] = -pow(b,3)*bar2B16/12+pow(b,5)*bar2A11*Ki/160
    S[1,5] = pow(b,5)*bar2A11*barD12*Ki/(360*barD22)
    S[1,6] = pow(b,5)*bar2A11*barB12*Ki/(360*barD22)

    S[2,2] = b*bar2D11
    S[2,4] = pow(b,3)*bar2B11/24-pow(b,5)*bar2A11*barD26*Ki/(180*barD22)+pow(b,7)*bar2A11*barB12*pow(Ki,2)/(10080*barD22)
    
    S[3,3] = pow(b,3)*bar2A11/12
    S[3,7] = -pow(b,5)*bar2A11*barB12/(720*barD22)

    S[4,4] = pow(b,5)*bar2A11/320
    S[4,6] = pow(b,5)*bar2A11*barB12/(720*barD22)
    S[4,8] = -pow(b,5)*bar2A11*barD26/(360*barD22)+ pow(b,7)*bar2A11*barB12*Ki/(10080*barD22)

    S[5,5] = pow(b,5)*bar2A11*pow(barD12,2)/(720*pow(barD22,2))
    S[5,6] = pow(b,5)*bar2A11*barD12*barB12/(720*pow(barD22,2))
    S[5,8] = -pow(b,5)*bar2A11*barD12*barD26/(360*pow(barD22,2))-pow(b,7)*bar2A11*barD12*barB12*Ki/(60480*pow(barD22,2))

    S[6,6]= pow(b,5)*bar2A11*pow(barB12,2)/(720*pow(barD22,2))
    S[6,8] = -pow(b,5)*bar2A11*barB12*barD26/(360*pow(barD22,2))-pow(b,7)*bar2A11*barB12*barB12*Ki/(60480*pow(barD22,2))

    S[7,7] = pow(b,7)*bar2A11*barB12*barB12/(10080*pow(barD22,2))-pow(b,7)*pow(bar2A11,2)/(30240*barD22)

    S[8,8] = pow(b,5)*bar2A11*pow(barD26,2)/(180*pow(barD22,2))+pow(b,5)*bar2A11*barD12/(360*barD22)+pow(b,7)*bar2A11*barB12*barD26*Ki/(15120*pow(barD22,2))-pow(b,9)*pow(bar2A11,2)*pow(Ki,2)/(90720*barD22)-pow(b,9)*bar2A11*pow(barB12,2)*pow(Ki,2)/(403200*barD22)
    for i in range(9):
        for j in range(i + 1, 9):
            S[j, i] = S[i, j]
        
    return S


# ---------------- Main Execution Example ---------------- 1
if __name__ == "__main__":
    Ki, Gamma11, k1, k2, k3 = 1, 2, 3, 2, 1 # Example values
    S_initial_9x9 = Crossection_9x9(Ki, Gamma11, k1, k2, k3, b, t)
    print("Initial 9x9 Cross-section Stiffness Matrix:")
    print(S_initial_9x9)
