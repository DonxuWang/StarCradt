import numpy as np
import argparse
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
 
 
if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--path', type=str, required=True)
    #args = parser.parse_args()
    path = r'C:\Users\Bea\Desktop\Project\StarCraft-master\result\qmix\3s5z\episode_rewards_0.npy'
    points = np.load(path)  # (n, 3)
 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
 
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='y')
    plt.show()