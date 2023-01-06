import argparse
import sys

import rospy
import sys
import baxter_interface

#sys.path.append(str('/home/developer/catkin_ws/src/baxter_common'))
#print(sys.path)
from baxter_interface import CHECK_VERSION
from baxter_core_msgs.msg import EndEffectorState, DigitalIOState

class Waypoints(object):
    def close_right_gripper(self,data):
        if data.state == DigitalIOState.PRESSED:
            self._right_gripper_state=False
            self._right_gripper.close()

    def open_right_gripper(self,data):
        if data.state == DigitalIOState.PRESSED:
            self._right_gripper_state=True
            self._right_gripper.open()

    def close_left_gripper(self,data):
        if data.state == DigitalIOState.PRESSED:
            self._left_gripper_state=False
            self._left_gripper.close()

    def open_left_gripper(self,data):
        if data.state == DigitalIOState.PRESSED:
            self._left_gripper_state=True
            self._left_gripper.open()
    def __init__(self, speed, accuracy):
        # Create baxter_interface limb instance
        #self._arm = limb
        self._left_limb = baxter_interface.Limb('left')
        self._right_limb = baxter_interface.Limb('right')

        #Create baxter_gripper interface

        self._left_gripper = baxter_interface.Gripper('left', CHECK_VERSION)
        self._right_gripper = baxter_interface.Gripper('right', CHECK_VERSION)
        self._left_gripper.set_holding_force(100)
        self._right_gripper.set_holding_force(100)

        # Parameters which will describe joint position moves
        self._speed = speed
        self._accuracy = accuracy

        # Recorded waypoints
        self._waypoints = list()

        # Recording state
        self._is_recording = False

        # Verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable()
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        
        # Create Navigator I/O
        self._left_navigator_io = baxter_interface.Navigator('left')
        self._right_navigator_io = baxter_interface.Navigator('right')

        #Ros Subscribe to gripper states
        self._left_gripper_state=True
        self._right_gripper_state=True
        rospy.Subscriber('/robot/digital_io/right_lower_button/state', DigitalIOState, self.close_right_gripper)
        rospy.Subscriber('/robot/digital_io/right_upper_button/state', DigitalIOState, self.open_right_gripper)
        rospy.Subscriber('/robot/digital_io/left_lower_button/state', DigitalIOState, self.close_left_gripper)
        rospy.Subscriber('/robot/digital_io/left_upper_button/state', DigitalIOState, self.open_left_gripper)

    def _record_waypoint(self, value):
        """
        Stores joint position waypoints
        Navigator 'OK/Wheel' button callback
        """
        
        if value:
            print("Waypoint Recorded")
            #LEFT RIGHT GripperLEFT GripperRight
            self._waypoints.append((self._left_limb.joint_angles(),self._right_limb.joint_angles(),self._left_gripper_state,self._right_gripper_state))
            print(self._waypoints)
            print("\n")

    def _stop_recording(self, value):
        """
        Sets is_recording to false
        Navigator 'Rethink' button callback
        """
        # On navigator Rethink button press, stop recording
        if value:
            self._is_recording = False

    def record(self):
        """
        Records joint position waypoints upon each Navigator 'OK/Wheel' button
        press.
        """
        rospy.loginfo("Waypoint Recording Started")
        print("Press Navigator 'OK/Wheel' button to record a new joint "
        "joint position waypoint.")
        print("Press Navigator 'Rethink' button when finished recording "
              "waypoints to begin playback")
        # Connect Navigator I/O signals
        # Navigator scroll wheel button press
        self._left_navigator_io.button0_changed.connect(self._record_waypoint)
        self._right_navigator_io.button0_changed.connect(self._record_waypoint)
        # Navigator Rethink button press
        self._left_navigator_io.button2_changed.connect(self._stop_recording)
        self._right_navigator_io.button2_changed.connect(self._stop_recording)

        # Set recording flag
        self._is_recording = True

        # Loop until waypoints are done being recorded ('Rethink' Button Press)
        while self._is_recording and not rospy.is_shutdown():
            rospy.sleep(1.0)

        # We are now done with the navigator I/O signals, disconnecting them
        self._left_navigator_io.button0_changed.disconnect(self._record_waypoint)
        self._right_navigator_io.button0_changed.disconnect(self._record_waypoint)
        self._left_navigator_io.button2_changed.disconnect(self._stop_recording)
        self._right_navigator_io.button2_changed.disconnect(self._stop_recording)

    def playback(self):
        #WRITE TO FILE
        filename=input("Enter file name: ")
        f=open(filename,"x")
        for way in self._waypoints:
            f.write(str(way))
            f.write("\n")
        f.close()
        """
        Loops playback of recorded joint position waypoints until program is
        exited
        """
        rospy.sleep(1.0)

        rospy.loginfo("Waypoint Playback Started")
        print("  Press Ctrl-C to stop...")

        # Set joint position speed ratio for execution
        self._left_limb.set_joint_position_speed(self._speed)
        self._right_limb.set_joint_position_speed(self._speed)

        # Loop until program is exited
        loop = 0
        while not rospy.is_shutdown():
            loop += 1
            print("Waypoint playback loop #%d " % (loop,))
            for waypoint in self._waypoints:
                if rospy.is_shutdown():
                    break
                self._left_limb.move_to_joint_positions(waypoint[0], timeout=20.0,threshold=self._accuracy)
                self._right_limb.move_to_joint_positions(waypoint[1], timeout=20.0,threshold=self._accuracy)
                if(waypoint[2]==True): #left gripper open
                    #print("OPEN L GRIPPER")
                    self._left_gripper.open()
                else:
                    #print("CLOSE L GRIPPER")
                    self._left_gripper.close()
                if(waypoint[3]==True): #right gripper open
                    #print("OPEN R GRIPPER")
                    self._right_gripper.open()
                else:
                    #print("CLOSE R GRIPPER")
                    self._right_gripper.close()

                # Sleep for a few seconds between playback loops
                rospy.sleep(1.0)
            rospy.sleep(2.0)

        # Set joint position speed back to default
        self._left_limb.set_joint_position_speed(0.3)
        self._right_limb.set_joint_position_speed(0.3)

    def clean_shutdown(self):
        print("\nExiting example...")
        if not self._init_state:
            print("Disabling robot...")
            self._rs.disable()
        return True


def main():
    """RSDK Joint Position Waypoints Example
    Records joint positions each time the navigator 'OK/wheel'
    button is pressed.
    Upon pressing the navigator 'Rethink' button, the recorded joint positions
    will begin playing back in a loop.
    """


    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    required = parser.add_argument_group('required arguments')
    #required.add_argument(
    #    '-l', '--limb', required=True, choices=['left', 'right'],
    #    help='limb to record/playback waypoints'
    #)
    parser.add_argument(
        '-s', '--speed', default=0.3, type=float,
        help='joint position motion speed ratio [0.0-1.0] (default:= 0.3)'
    )
    parser.add_argument(
        '-a', '--accuracy',
        default=baxter_interface.settings.JOINT_ANGLE_TOLERANCE, type=float,
        help='joint position accuracy (rad) at which waypoints must achieve'
    )
    args = parser.parse_args(rospy.myargv()[1:])


    print("Initializing node... ")
    rospy.init_node("rsdk_joint_position_waypoints_%s" % ('leftright',))
    #Calibrate Gripper--------------------------------------------------------
    print("Calibrating Grippers...")
    leftGripper = baxter_interface.Gripper('left', CHECK_VERSION)
    rightGripper = baxter_interface.Gripper('right', CHECK_VERSION)
    leftGripper.calibrate(*[])
    rightGripper.calibrate(*[])

    #-------------------------------------------------------------------------

    waypoints = Waypoints(args.speed, args.accuracy)

    # Register clean shutdown
    rospy.on_shutdown(waypoints.clean_shutdown)

    # Begin example program
    waypoints.record()
    waypoints.playback()

if __name__ == '__main__':
    main()
