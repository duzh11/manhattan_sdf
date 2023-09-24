import numpy as np
a=np.array([[3,2],[1,4]])
max_point=a.max(axis=0)
min_point=a.min(axis=0)
print(max_point)
print(min_point)