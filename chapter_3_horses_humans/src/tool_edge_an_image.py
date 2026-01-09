import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the test image
img = cv2.imread('test_horse.jpg', cv2.IMREAD_GRAYSCALE) # Grayscale the picture for simplicity

# 2. Define Vertical Edge Filter (like what is taught by Andrew Ng in his ML course)
# The filter emphasizes vertical edges and there is a different between left and right side
vertical_filter = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

#as the image resoltion is quite high, we can try bigger vertical and horizontal filters
vertical_5x5 = np.array([
    [-1, -2, 0, 2, 1],
    [-1, -2, 0, 2, 1],
    [-2, -4, 0, 4, 2],
    [-1, -2, 0, 2, 1],
    [-1, -2, 0, 2, 1]
])

horizontal_filter = np.array([[1, 2, 1],
                              [0, 0, 0],
                              [-1, -2, -1]])

#as the image resoltion is quite high, we can try bigger vertical and horizontal filters
horizontal_5x5 = np.array([
    [ 2,  4,  6,  4,  2],
    [ 1,  2,  3,  2,  1],
    [ 0,  0,  0,  0,  0],
    [-1, -2, -3, -2, -1],
    [-2, -4, -6, -4, -2]
])

sharpening_filter = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
size = 31
blurring_filter = np.ones((size, size), np.float32) / (size * size)


# 3. Do the convolution function (similar to tensorflow Conv2D)
vertical_edged_img = cv2.filter2D(img, -1, vertical_5x5)
horizontal_edged_img = cv2.filter2D(img, -1, horizontal_5x5)
sharpened_img = cv2.filter2D(img, -1, sharpening_filter)
blurred_img = cv2.filter2D(img, -1, blurring_filter)


# 4. Display the results

titles = ['Original', 'Vertical Edge', 'Horizontal Edge', 'Sharpen', 'Blur']
images = [img, vertical_edged_img, horizontal_edged_img, sharpened_img, blurred_img]

plt.figure(figsize=(25, 10)) # Width, Height in inches
for i in range(5):
    # divided to 2 rows, 3 columns
    plt.subplot(2, 3, i+1) 
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=18) # increase font size
    plt.axis('off')

plt.tight_layout() # show without overlapping
plt.show()