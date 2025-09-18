import numpy as np
import os

def load_data(loadfile, voxel=66):
    files = os.listdir(loadfile)
    # print(len(files))
    output = []
    epoch = 0

    for file in files:
        # load file
        file_dir = loadfile + '/' + file
        # num = [int(index) for index in file.split('.') if index.isdigit()][0]
        with open(file_dir, "r") as f:
            x = np.loadtxt(f, dtype="int")
            if len(x) == voxel**3:
                x = np.reshape(x, [voxel, voxel, voxel, 1])[1:-1,1:-1,1:-1].tolist()
                output.append(x)
        if len(output):
            np.save(r"/home/xiaoyang/PycharmProjects/pythonProject30_3d_foam/models/voxel/npy_0713_27b/%d.npy" % epoch, output)
            epoch += 1
            output = []
    # return np.array(output)


def main():
    loadfile = r"/home/xiaoyang/PycharmProjects/pythonProject30_3d_foam/models/voxel/0713_27b"
    x = load_data(loadfile)
    print(x.shape)



if __name__ == "__main__":
    main()