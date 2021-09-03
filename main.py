import numpy as np
import matplotlib.pyplot as plt
import cv2

#Vertical filter
vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
print(vertical_filter)

#Horizontal filter
horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]

#Load the pinwheel image from the personal computer
img=cv2.imread('C:/Users/Tuli/Desktop/pinwheel.jpg')
plt.imshow(img)
plt.show()


#Dimensions of the image
n,m,d = img.shape

#Edges image
edges_img = img.copy()

#Loop over all pixels in the image
for row in range(3, n-2):
    for col in range(3, m-2):

        #Create little local 3x3 box
        local_pixels = img[row-1:row+2, col-1:col+2, 0]

        #Vertical filter - apply
        vertical_transformed_pixels = vertical_filter*local_pixels
        #Remap the vertical score
        vertical_score = vertical_transformed_pixels.sum()/4

        #Horizontal filter - apply
        horizontal_transformed_pixels = horizontal_filter*local_pixels
        #Remap the horizontal score
        horizontal_score = horizontal_transformed_pixels.sum()/4

        #Combine the horizontal and vertical scores into a total edge score
        edge_score = (vertical_score**2 + horizontal_score**2)**.5

        #Insert this edge score into the edges image
        edges_img[row, col] = [edge_score]*3

#Remap the values in the 0-1 range
edges_img = edges_img/edges_img.max()
plt.imshow(edges_img)
plt.show()
