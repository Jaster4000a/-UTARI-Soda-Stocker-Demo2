#!/usr/bin/python2
import sys

import rospy
import numpy as np
import baxter_interface
from baxter_interface import CHECK_VERSION
import ast
from std_msgs.msg import String

filenames=("plastic_cup_2.txt","plastic_cup_1.txt")

def playback(data):
    can_array=data.data
    print(can_array)
    for n in range(len(can_array)):
        if can_array[n] == '1':
            if filenames[n] != "na":
                _waypoints=list()
                f=open(filenames[n],"r")
                for line in f:
                    waypointz=line[:-1].split('},')
                    gripper_states=ast.literal_eval("(" + waypointz[2])
                    w=(ast.literal_eval(waypointz[0][1:]+"}"),ast.literal_eval(waypointz[1][1:]+"}"),gripper_states[0],gripper_states[1])
                    _waypoints.append(w)

                f.close() 

                # Create baxter_interface limb instance
                #self._arm = limb
                _left_limb = baxter_interface.Limb('left')
                _right_limb = baxter_interface.Limb('right')

                #Create baxter_gripper interface

                _left_gripper = baxter_interface.Gripper('left', CHECK_VERSION)
                _right_gripper = baxter_interface.Gripper('right', CHECK_VERSION)
                _left_gripper.set_holding_force(100)
                _right_gripper.set_holding_force(100)

                _speed=1
                _accuracy=2*baxter_interface.settings.JOINT_ANGLE_TOLERANCE

                # Verify robot is enabled
                print("Getting robot state... ")
                _rs = baxter_interface.RobotEnable()
                _init_state = _rs.state().enabled
                print("Enabling robot... ")
                _rs.enable()

                rospy.sleep(1.0)

                rospy.loginfo("Waypoint Playback Started")

                # Set joint position speed ratio for execution
                _left_limb.set_joint_position_speed(_speed)
                _right_limb.set_joint_position_speed(_speed)

                for waypoint in _waypoints:
                    if rospy.is_shutdown():
                        break
                    _left_limb.move_to_joint_positions(waypoint[0], timeout=20.0,threshold=_accuracy)
                    _right_limb.move_to_joint_positions(waypoint[1], timeout=20.0,threshold=_accuracy)
                    if(waypoint[2]==True): #left gripper open
                        print("OPEN L GRIPPER")
                        _left_gripper.open()
                    else:
                        print("CLOSE L GRIPPER")
                        _left_gripper.close()
                    if(waypoint[3]==True): #right gripper open
                        print("OPEN R GRIPPER")
                        _right_gripper.open()
                    else:
                        print("CLOSE R GRIPPER")
                        _right_gripper.close()

                    # Sleep for a few seconds between playback loops
                    rospy.sleep(0.5)
                rospy.sleep(2.0)

                # Set joint position speed back to default
                _left_limb.set_joint_position_speed(0.3)
                _right_limb.set_joint_position_speed(0.3)
                print("Bye Bye")
        
if __name__ == '__main__':
    rospy.init_node('Playback_Subscriber',anonymous=True)
    # Initialze ROS node
    rospy.Subscriber('can_states', String, playback,queue_size=1)
    rospy.spin()