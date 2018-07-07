import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

data = np.load('model.npy')
ax = plt.subplot(111, projection="3d")
# ax = plt.figure().add_subplot(111, projection="3d")


ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
for d in data:
    x, y, z = d[0], d[1], d[2]
    ax.scatter(x, y, z, c="b")
plt.show()