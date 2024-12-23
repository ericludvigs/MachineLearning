import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy.random import normal, uniform

for i in range(0,2):
    # Load the terrain
    terrain = imread(f'SRTM_data_Norway_{i+1}.tif')

    N = 1000
    m = 5 # polynomial order
    terrain = terrain[:N,:N]
    # Creates mesh of image pixels
    x = np.linspace(0,1, np.shape(terrain)[0])
    y = np.linspace(0,1, np.shape(terrain)[1])
    x_mesh, y_mesh = np.meshgrid(x,y)

    z = terrain
    #X = create_X(x_mesh, y_mesh,m)


    # Show the terrain
    plt.figure()
    plt.title(f'Terrain over Norway {i+1}')
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
