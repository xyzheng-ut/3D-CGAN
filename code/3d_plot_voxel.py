import matplotlib.pyplot as plt
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    ax.grid(grid)
    ax.set_axis_off()
    return ax

filled = np.array([
    [[1, 0, 1], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 1], [1, 0, 0], [1, 0, 1]],
    [[1, 1, 0], [1, 1, 1], [0, 0, 0]]
])

ax = make_ax()



filedirection = "/home/xiaoyang/PycharmProjects/pythonProject30_3d_foam/cgan20220421/fake_image/154.npy"
data = np.squeeze(np.load(filedirection)).astype("int32")
data_pro = np.reshape(data,[16,-1])
data_pro = np.sum(data_pro,-1)/64**3
print(data_pro)

ax.voxels(data[1], facecolors='#1f77b430', edgecolors='gray')
plt.show()