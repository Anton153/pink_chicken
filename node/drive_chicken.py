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
from tensorflow.keras.models import load_model
import torch
import torchvision.transforms as transforms
from torch import nn
class DualOutputDrivingCNN(nn.Module):
    def __init__(self):
        super(DualOutputDrivingCNN, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),

            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.linear_velocity_head = nn.Linear(128, 1)
        self.angular_velocity_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.model(x)
        linear_velocity = self.linear_velocity_head(features)
        angular_velocity = self.angular_velocity_head(features)
        return linear_velocity, angular_velocity

class drive_robbie:

    def __init__(self):
        #init ros node
        rospy.init_node('move_robbie', anonymous=True)

        # load model &&&&&USED FOR .H5 FILETYPE&&&&&&&&&
        try:
            # model_path = "/home/fizzer/ENPH353_Competition/src/pink_chicken/driving_models/model_20241204_034748.h5"

            # def calibrated_softmax(logits, bias=1.5):
            #     boosted_logits = tf.concat([
            #         logits[:, 0:1] * bias,
            #         logits[:, 1:2],
            #         logits[:, 2:3] * bias
            #     ], axis=1)
            #     return tf.nn.softmax(boosted_logits)

            # # Load the model with the custom function
            # self.model = load_model(model_path, custom_objects={'calibrated_softmax': calibrated_softmax})

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Initialize and load the model
            self.model = DualOutputDrivingCNN()
            model_path = "/home/fizzer/ENPH353_Competition/pytorchTest_4.pth"

            checkpoint = torch.load(model_path, map_location=self.device)  # Load checkpoint to the correct device
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)  # Move the model to the correct device
            self.model.eval()  # Set the model to evaluation mode

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

        # Define image preprocessing transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        ])


    def image_processing_callback(self, data):
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # Debugging: Display the image (optional)
            cv.imshow("Image window", cv_image)
            cv.waitKey(3)

            # Preprocess the image for the PyTorch model
            input_image = self.transform(cv_image)  # Apply transformations
            input_image = input_image.unsqueeze(0)  # Add batch dimension
            input_image = input_image.to(self.device)  # Move to the same device as the model

            # Run prediction
            with torch.no_grad():
                predicted_linear, predicted_angular = self.model(input_image)

            # Map predictions to continuous values
            linear_velocity = predicted_linear.item()
            angular_velocity = predicted_angular.item()

            # Create and publish Twist message
            twist = Twist()
            twist.linear.x = linear_velocity
            twist.angular.z = angular_velocity

            rospy.loginfo(f"Predicted velocities - Linear: {linear_velocity:.2f}, Angular: {angular_velocity:.2f}")
            self.cmd_vel_pub.publish(twist) 

            # # Resize and normalize the image
            # input_image = cv.resize(cv_image, (128, 128))  # Match model's input size
            # input_image = input_image.astype('float32') / 255.0  # Normalize to [0, 1]
            # input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
            
            # # Predict probabilities (returns a list or tuple)
            # predicted_linear, predicted_angular = self.model.predict(input_image)
            
            # # Convert probabilities to discrete values
            # linear_velocity = np.argmax(predicted_linear) - 1  # Map 0->-1, 1->0, 2->1
            # angular_velocity = np.argmax(predicted_angular) - 1  # Map 0->-1, 1->0, 2->1
            
            # # Create and publish Twist message
            # twist = Twist()
            # twist.linear.x = float(linear_velocity) * 1.72  # Convert to float to match ROS message type
            # twist.angular.z = float(angular_velocity) * 1.72
            
            # rospy.loginfo(f"Predicted velocities - Linear: {linear_velocity}, Angular: {angular_velocity}")
            # self.cmd_vel_pub.publish(twist)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
        except Exception as predict_error:
            rospy.logerr(f"Error in prediction: {predict_error}")
    
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

