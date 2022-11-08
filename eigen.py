import numpy as np
import time as t

def submatrix(mat, rowdel, coldel):
    matSub = np.zeros([len(mat)-1, len(mat)-1]);
    rowSub = 0;
    for i in range(len(mat)):
        colSub = 0;
        for j in range(len(mat[0])):
            if (i != rowdel and j != coldel):
                matSub[rowSub][colSub] = mat[i][j];
                
                colSub += 1;

                if (colSub == len(mat[0])-1):
                    rowSub += 1;
                    colSub = 0;
    return matSub;

def determinan(mat):
    det = 0;
    sign = 1; 
    if (len(mat) == 1):
        return mat[0][0];

    kof = np.zeros([len(mat)-1, len(mat)-1]);

    for i in range(len(mat)):
        kof = submatrix(mat, 0, i);
        det += (sign * mat[0][i] * determinan(kof));

        sign *= -1;

    return det; 

M = np.array([ [3, -2, 0],
                [2, 3, 0],
                [0, 0, 5],]);

start = t.time();
print(determinan(M));
# count = 9999999999999;
# for i in range(100):
#     count += i;
#     print(count);
print("--- %s seconds ---" % (t.time() - start));

