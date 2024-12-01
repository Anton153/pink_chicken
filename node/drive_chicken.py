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
import tensorflow as tf

class drive_robbie:

    def __init__(self):
        #init ros node
        rospy.init_node('move_robbie', anonymous=True)

        #load model
        try:
            model_path = "/home/fizzer/ENPH353_Competition/src/pink_chicken/driving_models/best_model.h5"
            self.model = tf.keras.models.load_model(model_path)
            rospy.loginfo("Model loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Error loading model: {e}")
            sys.exit(1)

        #init cv bridge 
        self.bridge = CvBridge()

        # SUBSCRIPTIONS 
        self.image_sub = rospy.Subscriber("rrbot/camera1/image_raw", Image, self.image_processing_callback)
        self.clock_sub = rospy.Subscriber('/clock', Clock, self.clock_callback) #write clock_callback

        # PUBLISHERS
        self.score_pub = rospy.Publisher('/score_tracker', String, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size = 10)

    def image_processing_callback(self,data):
        print("Image processing callback")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            print("Image received")
        except CvBridgeError as e:
            print(e)
        cv.imshow("Image window", cv_image)
        cv.waitKey(3)

        #drive with model
        input_image = cv.resize(cv_image, (112, 112))
        input_image = np.expand_dims(input_image, axis=0) #add batch dimension

        #predict 
        print("predicing velocities")
        predicted_velocities = self.model.predict(input_image)
        print(predicted_velocities)
        linear_velocity, angular_velocity = predicted_velocities[0]

        #publish 
        twist = Twist()
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity
        print("publishing....")
        self.cmd_vel_pub.publish(twist)
    
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

