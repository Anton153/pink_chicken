#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Int32
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from std_msgs.msg import String


# Global Variables
latest_time = None
elapsed_time = 0
robot_moving = False
# Set to track already processed contours
processed_contours = []

def clock_callback(data):
    '''
    Callback for /clock topic
    Keeps track of latest time for simulation
    '''

    global latest_time
    latest_time = data.clock
    # print(latest_time)

def camera_callback(data):
    '''
    Callback function for camera
    '''
    print("ESTABLISH CAMERA CALLBACK")
    cmd_vel_pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size = 10)
    bridge = CvBridge()
    try: 
        image = bridge.imgmsg_to_cv2(data, 'bgr8')
    except CvBridgeError as e:
        print(e)
    
    # print('####### Image received #######')
    #find command velocity
    # cx, width = det_course(image)
    # angular_z = PID(cx, width)

    # signSpotted = False 
    # signSpotted = find_sign(image)
    # if signSpotted:
    #     print('#########SIGN SPOTTED#########')

    # #move with PID
    # twist = Twist()
    # if angular_z == -1:
    #     twist.linear.x = -0.1
    #     twist.angular.z = 0.0
    #     cmd_vel_pub.publish(twist)
    #     return
    # else:
    #     twist.linear.x = 0.3
    #     twist.angular.z = angular_z
    #     cmd_vel_pub.publish(twist)
    #     return
# def PID(cx, width): 
    # PID controller
    # if cx != -1:
    #     error = cx - width // 2
    #     P = -0.01
    #     I = 0.01 
    #     D = 0.01

    #     # integral += I * error
    #     proportional = P * error
    #     output = proportional
    #     return output
    # else:
    #     return -1 #error code for no contours found


    # derivative = D * (error )

# def reset_integral():
#     global integral
#     integral = 0

# def adaptive_closing_and_dilation(road_mask, proximity):
    """
    Perform adaptive morphological operations based on the robot's proximity to separating lines.
    """
    # # Adjust kernel size based on proximity (closer = larger kernel)
    # base_size = 5
    # kernel_size = base_size + int(proximity / 10)  # Scale with proximity
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))

    # # Apply closing and dilation
    # closed_mask = cv.morphologyEx(road_mask, cv.MORPH_CLOSE, kernel)
    # dilated_mask = cv.dilate(closed_mask, kernel, iterations=1)
    # return dilated_mask

#def apply_perspective_mask(mask, height):
    """
    Apply a perspective mask to focus on the lower (closer) portion of the image.
    """
    # mask_height, mask_width = mask.shape
    # perspective_mask = np.zeros_like(mask, dtype=np.uint8)
    # cv.rectangle(perspective_mask, (0, mask_height // 3), (mask_width, mask_height), 255, -1)
    # return cv.bitwise_and(mask, perspective_mask)

#def det_course(image):
    '''
    Function that determines velocity based on image
    '''
    height, width = image.shape[:2]
    cx = width // 2  # Default to the center if no contours are found
    margin = int(0.0 * width)  # Calculate 15% of the width
    cropped_image = image[height // 2:, margin:width - margin]
    gray = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9, 9), 0)
    # median = cv.medianBlur(gray, 5)  # 5 is the kernel size

    # # Threshold for the dark road color
    # lower_bound = 79  
    # upper_bound = 85  
    # _, lower_thresh = cv.threshold(blurred, lower_bound, 255, cv.THRESH_BINARY)
    # _, upper_thresh = cv.threshold(blurred, upper_bound, 255, cv.THRESH_BINARY_INV)
    # road_mask = cv.bitwise_and(lower_thresh, upper_thresh)

    # road_mask = apply_perspective_mask(road_mask, height)
    # proximity = 100  # Example: Adjust this based on robot-camera distance logic
    # road_mask = adaptive_closing_and_dilation(road_mask, proximity)

    # cv.imshow('lines', road_mask)
    
    # contours, _ = cv.findContours(road_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # if contours:
    #     # print("Contours found")
    #     # Find the largest contour
    #     largest_contour = max(contours, key=cv.contourArea)

    #     # Calculate moments of the largest contour
    #     moments = cv.moments(largest_contour)

    #     if moments["m00"] != 0:
    #         # print("Moments found")
    #         # Calculate the center of mass of the largest contour
    #         cx = int(moments["m10"] / moments["m00"])
    #         cy = int(moments["m01"] / moments["m00"])
            
    #         # Adjust centroid coordinates to the original image
    #         cy += height // 2  # Offset y-coordinate since we cropped the lower half
            
    #         # Step 6: Visualize the results
    #         # Draw the largest contour on the original image
    #         cv.drawContours(image, [largest_contour + np.array([[0, height // 2]])], -1, (0, 255, 0), 2)
    #         # Draw a red circle at the centroid
    #         cv.circle(image, (cx, cy), 10, (0, 0, 255), -1)
    # else:
    #     print("No contours found")
    #     return -1, -1 #error code for no contours found
    # cv.imshow('bw', image) #check that it works
    # cv.waitKey(1)

    # return cx, width

#def find_sign(image):
    '''
    Finds signs in the image, which all have blue borders
    '''
    global processed_contours

    # # Convert image to HSV and identify blue regions
    # hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # low_blue = np.array([100, 150, 50])
    # high_blue = np.array([140, 255, 255])
    # mask = cv.inRange(hsv, low_blue, high_blue)

    # # Blur and detect edges
    # blurred = cv.GaussianBlur(mask, (5, 5), 0)
    # edges = cv.Canny(blurred, 50, 150)

    # # Find contours in the edge-detected image
    # contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # valid_contours = []

    # # Draw contours and check for retriggering
    # contours_image = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    # for contour in contours:
    #     # Get bounding box of the contour
    #     x, y, w, h = cv.boundingRect(contour)

    #     # Check if this contour is already processed
    #     is_duplicate = False
    #     for px, py, pw, ph in processed_contours:
    #         if abs(px - x) < 10 and abs(py - y) < 10:  # Adjust tolerance as needed
    #             is_duplicate = True
    #             break

    #     if not is_duplicate:
    #         # Mark the contour as processed
    #         processed_contours.append((x, y, w, h))

    #             # Check the boundaries of the edge image directly
    #         if np.any(edges[:, 0] == 255) and np.sum(edges[:, 0] == 255) > 10:
    #             print("Sign on the left spotted")  # Left edge
    #             # Draw this contour for visualization
    #             cv.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Green contours
    #             valid_contours.append(contour)
    #             return True 
    #         elif np.any(edges[:, -5] == 255) and np.sum(edges[:, -5:] == 255) > 10:
    #             print("Sign on the right spotted")  # Right edge
    #             # Draw this contour for visualization
    #             cv.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Green contours
    #             return True, valid_contours.append(contour)
    #         else:
    #             return False 
    # cv.imshow('signs', image)
    # cv.waitKey(1)


def move_robbie():
    '''
    Callback function that processes image and controls robot
    '''

    global elapsed_time, robot_moving

    print('running move_robbie')


    # Publishers
    score_pub = rospy.Publisher('/score_tracker', String, queue_size=1)
    # Subscribers
    rospy.Subscriber('/clock', Clock, clock_callback)
    rospy.Subscriber('rrbot/camera1/image_raw', Image, camera_callback)

    rospy.loginfo("Move robbie has started")
    rospy.sleep(1.0) #sleep for 1 second before doing stuff

    # Start the timer
    start_timer = String()
    # Start the timer
    print("publishing start timer")
    score_pub.publish('ElGato,kebab,0,NAAA')  # Adjust team name and password
    rospy.sleep(2)
    print("setting rate")
    rate = rospy.Rate(10)  # 10Hz loop

    while not rospy.is_shutdown():
        # Check if the timer exceeds the threshold
        if latest_time and latest_time.to_sec() > 10000:  # Adjust threshold if necessary
            end_timer = String()
            end_timer.data = 'ElGato,kebab,-1,NA'
            rospy.loginfo("Stopping the timer: Time exceeded 10 seconds")
            score_pub.publish(end_timer)  # Stop the timer
            break

        rate.sleep()


if __name__ == '__main__':
    # Initialze ROS node
    rospy.init_node('move_robbie', anonymous=True)
    try:
        move_robbie()
    except rospy.ROSInterruptException:
        rospy.on_shutdown(cv.destroyAllWindows)
        pass