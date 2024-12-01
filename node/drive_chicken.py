#!/usr/bin/env python3

from __future__ import print_function
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Int32
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from std_msgs.msg import String
import sys

class drive_robbie:

    def __init__(self):
        #init ros node
        rospy.init_node('move_robbie', anonymous=True)

        #init cv bridge 
        self.bridge = CvBridge()

        # SUBSCRIPTIONS 
        self.image_sub = rospy.Subscriber("rrbot/camera1/image_raw", Image, self.image_processing_callback)
        self.clock_sub = rospy.Subscriber('/clock', Clock, self.clock_callback) #write clock_callback

        # PUBLISHERS
        self.score_pub = rospy.Publisher('/score_tracker', String, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size = 10)

    def image_processing_callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv.imshow("Image window", cv_image)
        cv.waitKey(3)

        #drive. Called in image callback to drive based on image processing
        #self.drive()
    
    def clock_callback(self,data):
        '''
        Callback for /clock topic
        Keeps track of latest time for simulation
        '''

        global latest_time
        latest_time = data.clock
    
    def drive(self):
        twist = Twist()
        twist.linear.x = 0.3
        self.cmd_vel_pub.publish(twist)

def main(args):
    ic = drive_robbie()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()

if __name__ == '__main__':  
    main(sys.argv)  

