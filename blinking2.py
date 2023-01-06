#!/usr/bin/env python
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append("/home/developer/anaconda3/lib/python3.6/site-packages")
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import rospy
from baxter_interface import Limb
import baxter_interface
import time
import cv2
from cv_bridge import CvBridge
import rospkg
from sensor_msgs.msg import Image
import random
import numpy as np



def callback(data):
	global msg
	global msg_closed
	if not data == msg_closed:
		msg = data

if __name__ == '__main__':
	rospy.init_node('blinking')

	_images = ''#'/home/scem/ros_ws/baxter_eyes'

	img = cv2.imread(_images + 'straight.jpg')
	sub = rospy.Subscriber('/robot/xdisplay', Image, callback)

	bridge = CvBridge()
	img_closed = cv2.imread('closed.jpg')
	msg_closed = bridge.cv2_to_imgmsg(img_closed)#np.array(img_closed)
	msg = bridge.cv2_to_imgmsg(img)

	pub = rospy.Publisher('/robot/xdisplay', Image,latch=True, queue_size=10)

	while not rospy.is_shutdown():
		
		timer = random.randint(2,4)	

		time.sleep(timer)
		print("blinking")
		pub.publish(msg_closed)
		pub.publish(msg)