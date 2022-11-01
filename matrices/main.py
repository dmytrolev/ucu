import numpy
from scipy.linalg import blas
import time

N = int(input())

A = []
for r in range(N):
    A.append([int(a) for a in input().split(" ")])

B = []
for r in range(N):
    B.append([int(a) for a in input().split(" ")])

An = numpy.array(A)
Bn = numpy.array(B)

start = time.perf_counter()
Cn = blas.dgemm(1.0, An, Bn)
end = time.perf_counter()

print(end - start)
