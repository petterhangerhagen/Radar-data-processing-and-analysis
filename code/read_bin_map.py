import numpy as np
import matplotlib.pyplot as plt
import math

# data = np.fromfile("/home/aflaptop/catkin_ws/src/ros_af_shapefiles/data/land_maps/ravnkloa_radar_base_map.bin",dtype=np.float32)
# print(np.shape(data))


# # Reshape the data into a 2D array with 100 rows (adjust as necessary)
# # The '-1' tells numpy to calculate the size of this dimension based on the size of the array
# data = data.reshape((100, -1))

# # Plot the data
# plt.imshow(data, cmap='hot', interpolation='nearest')
# plt.show()

# libraries
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

# Set the plot size for this notebook:
plt.rcParams["figure.figsize"]=13,13

# Always start witht the basemap function to initialize a map
# m=Basemap(llcrnrlon=-100, llcrnrlat=-58,urcrnrlon=-30,urcrnrlat=15)
#m=Basemap(llcrnrlon=63.43277239967358, llcrnrlat=10.388008064091254,urcrnrlon=63.43683222708964,urcrnrlat=10.396477338245319)
# 63.43683222708964, 10.396477338245319
# 63.43277239967358, 10.388008064091254
# Show the coast lines

# m = Basemap(projection='tmerc', 
#             lat_0=63.4325,
#             lon_0=10.4012,
#             width=200000,
#             height=200000,
#             resolution='i')


# m.drawcoastlines()
 
# plt.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the image
img = mpimg.imread('/home/aflaptop/Documents/radar_tracker/data/map_crop.png')

# Create a subplot
fig, ax = plt.subplots()

# Display the image in the subplot
ax.imshow(img, extent=[0, 10, 0, 10])

# Add a scatter plot on top of the image
ax.scatter([2, 4, 6, 8], [3, 5, 7, 9])

# Display the plot
plt.show()
