#!/usr/bin/python2
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append("/home/developer/anaconda3/lib/python3.6/site-packages")
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image as IMAGE
import baxter_interface
from baxter_interface import CHECK_VERSION
from std_msgs.msg import String
import tkinter
from tkinter import *
import PIL.Image, PIL.ImageTk
import time
import math

RED_DEFAULT=237
GREEN_DEFAULT=158
BLUE_DEFAULT=174

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 0
        #Input RGB Value 
        self.red_value = 237
        self.green_value = 158
        self.blue_value = 174
        self.selected_color=(self.blue_value,self.green_value,self.red_value)
        #Input lower and upper limits
        self.lowerLimit=np.array((87,67,136))
        self.upperLimit=np.array((197,181,255))
        
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture()

        #Set up widgets
        self.left_label=Label(self.window,text="left bound")
        self.left_label.pack(side=tkinter.TOP)
        self.left_bound_slider=Scale(self.window,from_=0,to=self.vid.width/2,orient=HORIZONTAL)
        self.left_bound_slider.pack(side=tkinter.TOP)
        self.right_label=Label(self.window,text="right bound")
        self.right_label.pack(side=tkinter.TOP)
        self.right_bound_slider=Scale(self.window,from_=self.vid.width/2,to=self.vid.width,orient=HORIZONTAL)
        self.right_bound_slider.pack(side=tkinter.TOP)
        self.upper_label=Label(self.window,text="upper bound")
        self.upper_label.pack(side=tkinter.LEFT)
        self.upper_bound_slider=Scale(self.window,from_=0,to=self.vid.height,orient=VERTICAL)
        self.upper_bound_slider.set(self.vid.height/4)
        self.upper_bound_slider.pack(side=tkinter.LEFT)
        self.lower_label=Label(self.window,text="lower bound")
        self.lower_label.pack(side=tkinter.LEFT)
        self.lower_bound_slider=Scale(self.window,from_=0,to=self.vid.height,orient=VERTICAL)
        self.lower_bound_slider.pack(side=tkinter.LEFT)
        self.lower_bound_slider.set(self.vid.height/2)
        self.sections_label=Label(self.window,text="Num of Divisions")
        self.sections_label.pack(side=tkinter.RIGHT)
        self.section_divisions=Entry(self.window,width=10)
        self.section_divisions.pack(side=tkinter.RIGHT)
        self.section_divisions.insert(0,"1")
        self.enter_divisions=Button(self.window,text="Enter",command=self.divisions_callback)
        self.enter_divisions.pack(side=tkinter.RIGHT)
        self.divisions_value=1
        # Color Picker Button UNUSED
        #self.start_color_pick=Button(self.window,text="Color Picker",command=self.color_picker_dialog_callback)
        #self.start_color_pick.pack(side=tkinter.BOTTOM)
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        #self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        #self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def divisions_callback(self): #activates when button is pressed to confirm number of divisions
        self.divisions_value=self.section_divisions.get()
    ''' # UNUSED
    def color_picker_dialog_callback(self): 
         Get a frame from the video source
        ret, frame = self.vid.get_frame_clean()
        if ret:
            self.img=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("color_picker_window", self.img)
            self.color_explore=np.zeros((150,150,3),np.uint8)
            cv2.setMouseCallback("color_picker_window",self.color_selected_callback,param=self.img)
    '''
    '''# UNUSED                
    def color_selected_callback(self,event,x,y,flags,param):
        B=self.img[y,x][0]
        G=self.img[y,x][1]
        R=self.img[y,x][2]
        self.color_explore [:] = (B,G,R)
        #print("Explore: " + str(self.color_explore))
        #print("BGR: %d %d %d"%(B,G,R))
        cv2.imshow("eyeDropper",self.color_explore)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_color=np.uint8([[[B,G,R]]])
            print("COLOR: "+str(self.selected_color))
            hsvColor=cv2.cvtColor(self.selected_color,cv2.COLOR_BGR2HSV)
            self.lowerLimit=np.array((hsvColor[0][0][0] - 10, 100, 100))
            self.upperLimit=np.array((hsvColor[0][0][0] + 10, 255, 255))
            self.selected_color=(B,G,R)
            cv2.destroyAllWindows()
    '''
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame(self.selected_color,self.lowerLimit,self.upperLimit,self.left_bound_slider.get(),self.right_bound_slider.get(),self.upper_bound_slider.get(),self.lower_bound_slider.get(),int(self.divisions_value))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)

class MyVideoCapture: 
    #Obtain Camera Feed and Draw Bounding Lines
    def __init__(self, video_source=0):
        rospy.init_node('Camera_Subscriber',anonymous=True)
        print("Calibrating Grippers...")
        leftGripper = baxter_interface.Gripper('left', CHECK_VERSION)
        rightGripper = baxter_interface.Gripper('right', CHECK_VERSION)
        leftGripper.calibrate(*[])
        rightGripper.calibrate(*[])
        self.height=800
        self.width=1280
        # Subscribe to head_camera image topic
        rospy.Subscriber('/cameras/head_camera/image', IMAGE, self.image_callback)
        # Publish to can_states
        self.can_pub=rospy.Publisher('can_states',String,queue_size=10)
        self.can_string=list("0")
        self.vid=np.empty(1)#np.array((100,100,100))
        self.vid_clean=np.empty(1)


    def image_callback(self,image_data): 
        # Obtain camera feed from subscriber
        self.vid=np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
        self.height=image_data.height
        self.width=image_data.width

    def get_frame(self,selected_color,lower_limit,upper_limit,left_line_bound,right_line_bound,upper_line_bound,lower_line_bound,num_of_divisions):
        # Draw bounding boxes on camera feed

        frame = self.vid
        ret = True
        
        cv_image = frame

        hsv_im = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_im, lower_limit, upper_limit)

        kernel = np.ones((7,7), np.uint8)

        filtered_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        filtered_mask_2 = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)

        segmented_im = cv2.bitwise_and(cv_image, cv_image, mask=filtered_mask_2)

        # Draw a boundary of the detected objects
        contours, heirarchy = cv2.findContours(filtered_mask_2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 10 and area < 500):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.line(frame,(left_line_bound,lower_line_bound),(left_line_bound,upper_line_bound),(0,0,255),5)
                cv2.line(frame,(right_line_bound,lower_line_bound),(right_line_bound,upper_line_bound),(0,0,255),5)
                if num_of_divisions>1:
                    self.can_string=['0']*num_of_divisions
                    for n in range(1,num_of_divisions+1):
                        sub_div=math.floor(left_line_bound+(right_line_bound-left_line_bound)*(n/num_of_divisions))
                        sub_div_minus=math.floor(left_line_bound+(right_line_bound-left_line_bound)*((n-1)/num_of_divisions))
                        if n < num_of_divisions:
                            cv2.line(frame,(sub_div,lower_line_bound),(sub_div,upper_line_bound),(255,0,0),5)
                        if (upper_line_bound<y<lower_line_bound) or (lower_line_bound<y<upper_line_bound):
                            if sub_div_minus<x<sub_div:
                                self.can_string[n-1]='1'
                                imageFrame = cv2.rectangle(cv_image, (x, y),
                                        (x + w, y + h),
                                        (0,0,255), 2)
                
                                cv2.putText(cv_image, "Restock " + str(n), (x, y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0,0,255))
                            #self.can_string[n-1]='0'
                        #self.can_string[n-1]='0'
                elif num_of_divisions==1:
                    self.can_string[0]='0'
                    if (upper_line_bound<y<lower_line_bound) or (lower_line_bound<y<upper_line_bound):
                        if left_line_bound<x<right_line_bound:
                            self.can_string[0]='1'
                            imageFrame = cv2.rectangle(cv_image, (x, y),
                                        (x + w, y + h),
                                        (0,0,255), 2)
                
                            cv2.putText(cv_image, "Restock 0", (x, y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0,0,255))
            print("".join(self.can_string))
            self.can_pub.publish("".join(self.can_string))
            
        if ret:
            # Return a boolean success flag and the current frame converted to BGR
            return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            return (ret, None)
    #else:
    #    return (ret, None)
''' #UNUSED
    def get_frame_clean(self):
        frame=self.vid
        ret=True
        #if self.vid.isOpened():
        #    ret, frame = self.vid.read()
        #    if ret:
        #        # Return a boolean success flag and the current frame converted to BGR
        return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #    else:
        #        return (ret, None)
        #else:
        #    return (ret, None)

    # Release the video source when the object is destroyed
    #def __del__(self):
    #    if self.vid.isOpened():
    #        self.vid.release
'''
App(tkinter.Tk(),"Tkinter and OpenCV")