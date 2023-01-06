# import required libraries
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append("/home/developer/anaconda3/lib/python3.6/site-packages")
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
# define a numpy.ndarray for the color
# here insert the bgr values which you want to convert to hsv
color = np.uint8([[[57,88,99]]])

# convert the color to HSV
hsvColor = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

# display the color values
print("BGR of color:", color)
print("HSV of color:", hsvColor)

# Compute the lower and upper limits
lowerLimit = hsvColor[0][0][0] - 10, 100, 100
upperLimit = hsvColor[0][0][0] + 10, 255, 255

# display the lower and upper limits
print("Lower Limit:",lowerLimit)
print("Upper Limit", upperLimit)
