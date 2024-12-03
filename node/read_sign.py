#!/usr/bin/env python3


from __future__ import print_function
from os import wait

import roslib
import sys
import rospy
import cv2
import numpy as np
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
class read_sign(QtWidgets.QMainWindow):
    """
    A Qt GUI application integrated with ROS to display:
    - A reference image in label 1 (left).
    - A live feed from the Gazebo camera with SIFT object detection in label 2 (right).
    """

    def __init__(self):
        # Initialize the Qt parent class
        super(read_sign, self).__init__()

        # Initialize ROS node
        rospy.init_node('read_sign', anonymous=True)

        # loadUi("/home/fizzer/ros_ws/src/pink_chicken/node/SIFT_robot.ui", self)


        # CvBridge for converting ROS images to OpenCV
        self.bridge = CvBridge()
        self.latest_frame = None  # Placeholder for the live feed frame

        # Load the reference image
        self.reference_image = cv2.imread("/home/fizzer/ros_ws/src/pink_chicken/node/blank.jpg")

        # pixmap = self.convert_cv_to_pixmap(self.reference_image)
        # self.label.setPixmap(pixmap)
        # self.label.setScaledContents(True)
        self.max_grey = 0
        self.best_clueboard = None
        self.no_blue_counter = 0

        # ROS Subscriber for the Gazebo camera
        self.image_sub = rospy.Subscriber("rrbot/camera1/image_raw", Image, self.callback)

        # Timer to periodically update the GUI and process the live feed
        self.query_timer = QtCore.QTimer(self)
        self.query_timer.timeout.connect(self.SLOT_query_camera)
        self.query_timer.start(1) 


    def callback(self, data):
        """
        Callback function to process incoming ROS camera feed.
        """
        try:
            # Convert ROS image message to OpenCV format
            self.latest_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Error converting image: {e}")

    def convert_cv_to_pixmap(self, cv_img):
        """
        Converts an OpenCV image (BGR format) to QPixmap for displaying in a QLabel.
        """
        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width

        # Create QImage from the RGB image
        qimage = QtGui.QImage(rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

        # Convert QImage to QPixmap
        return QtGui.QPixmap.fromImage(qimage)
    

    def reset_grey_count(self):
        """
        Reset the highest grey-pixel counter as we move to next board
        """
        # rospy.loginfo("Resetting grey pixel measurement.")
        self.max_grey = 0  # Reset maximum grey percentage
        self.best_clueboard = None  # Clear the stored best clueboard
        self.no_blue_counter = 0  # Reset the no blue detection counter


    
    def SLOT_query_camera(self):
        """
        Processes the live feed using SIFT keypoint matching to visualize matches
        and extract the clueboard, enhanced with blue border detection.
        """
        if self.reference_image is None:
            rospy.logwarn("No reference image loaded. Please select a reference image first.")
            return

        if self.latest_frame is None:
            rospy.logwarn("No live feed frame available yet.")
            time.sleep(10)
            return

        # Detect blue-bordered clueboard
        blue_roi, blue_corners, blue_corners_adjusted = self.detect_blue_border()
        if blue_roi is None:
            # rospy.logwarn("No blue border detected.")
            self.no_blue_counter += 1
            if self.no_blue_counter >= 5:
                self.reset_grey_count()
            return
        if blue_roi is not None:
            clueboard_transformed = self.straighten_from_grey_clueboard(blue_roi)
            if clueboard_transformed is not None:
                # cv2.imshow("Final Clueboard", clueboard_transformed)
                # cv2.waitKey(1)
                cropped = self.process_image(clueboard_transformed)
                self.cnn(cropped)
        # Process SIFT keypoint matching
        good_matches, kp_ref, kp_live, adjusted_kp_live, gray_live_frame = self.perform_sift_matching(blue_roi)
        if good_matches is None:
            rospy.logwarn("No valid matches found.")
            return

        # Visualize matches
        # self.visualize_matches(good_matches, kp_ref, kp_live, adjusted_kp_live)

        # Perform homography and straighten the clueboard
        # self.straighten_clueboard(good_matches, kp_ref, kp_live, blue_corners, blue_corners_adjusted, blue_roi)

    def padding(self, cropped, target_size):
        """
        Resizes an image to the target size by padding it with black pixels.
        """
        target_width, target_height = target_size
        h, w = cropped.shape[:2]

        # Calculate scaling factor to preserve aspect ratio
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize the image using the calculated scale
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Calculate padding to center the resized image
        pad_left = (target_width - new_w) // 2
        pad_right = target_width - new_w - pad_left
        pad_top = (target_height - new_h) // 2
        pad_bottom = target_height - new_h - pad_top

        # Add black padding
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # print("Padded image shape:", padded.shape)
        return padded

    def process_image(self, clueboard_transformed):
        hsv_clueboard = cv2.cvtColor(clueboard_transformed, cv2.COLOR_BGR2HSV)
        lower_blue, upper_blue = np.array([100, 120, 70]), np.array([150, 255, 255])
        blue_clue = cv2.inRange(hsv_clueboard, lower_blue, upper_blue)
        clue_copy = clueboard_transformed.copy()


        # Define rectangle dimensions
        rect_width = 49  # Width of each rectangle
        rect_height = 100  # Height of each rectangle

        # Define the starting position for the first character (x, y)
        top_start = (265, 40)  # Starting point
        bot_start = (30,310)
       
        rectangles = []


        for i in range(8):
            top_left = (top_start[0] + i * rect_width, top_start[1])
            bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)
            rectangles.append((top_left, bottom_right))

        # Generate rectangles for the bottom row (B0 to B9)
        for i in range(12):
            top_left = (bot_start[0] + i * rect_width, bot_start[1])
            bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)
            rectangles.append((top_left, bottom_right))


        for (top_left, bottom_right) in rectangles:
            cv2.rectangle(clue_copy, top_left, bottom_right, (0, 255, 0), 1)  # Green rectangles, thickness=1
            cv2.imshow("rectangles", clue_copy)
            cv2.waitKey(1)

        cropped_images = []

        for (top_left, bottom_right) in rectangles:
            x1, y1 = top_left
            x2, y2 = bottom_right

            cropped = blue_clue[y1:y2, x1:x2]
            if cv2.countNonZero(cropped) >=200:
                cropped = self.padding(cropped, target_size=(100, 150))
                cropped_images.append(cropped)
        for i, cropped in enumerate(cropped_images):
            cv2.imshow(f"Cropped {i+1}", cropped)
            cv2.waitKey(1)
        return cropped_images


    def detect_blue_border(self):
        """
        Detects the largest blue-bordered quadrilateral in the live frame.
        """
        hsv_frame = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2HSV)
        lower_blue, upper_blue = np.array([100, 120, 70]), np.array([150, 255, 255])
        blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area, blue_corners = 0, None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4 and area > max_area:
                    max_area = area
                    blue_corners = approx.reshape(4, 2)

        if blue_corners is not None:
            x, y, w, h = cv2.boundingRect(blue_corners)
            blue_roi = self.latest_frame[y:y + h, x:x + w]
            blue_corners_adjusted = blue_corners - [x, y]
            
            # Check if blue_roi contains red pixels
            hsv_roi = cv2.cvtColor(blue_roi, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 50, 50])   # Lower bound for red hue
            upper_red1 = np.array([10, 255, 255])  # Upper bound for red hue
            lower_red2 = np.array([170, 50, 50])  # Wrap-around for red hue
            upper_red2 = np.array([180, 255, 255])

            # Create masks for red color
            red_mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            # Check if red pixels are present
            if cv2.countNonZero(red_mask) > 0:
                return blue_roi, blue_corners, blue_corners_adjusted

        return None, None, None

    def perform_sift_matching(self, blue_roi):
        """
        Detects and matches SIFT keypoints between the reference image and blue ROI.
        """
        gray_ref_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
        gray_live_frame = cv2.cvtColor(blue_roi, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("blueROI", blue_roi)

        sift = cv2.SIFT_create()
        kp_ref, desc_ref = sift.detectAndCompute(gray_ref_image, None)
        kp_live, desc_live = sift.detectAndCompute(gray_live_frame, None)

        if desc_ref is None or desc_live is None:
            return None, None, None, None, None

        flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), {})
        matches = flann.knnMatch(desc_ref, desc_live, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]
        adjusted_kp_live = [cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                            for kp in kp_live]
    
        return good_matches, kp_ref, kp_live, adjusted_kp_live, gray_live_frame

    def straighten_clueboard(self, good_matches, kp_ref, kp_live, blue_corners, blue_corners_adjusted, blue_roi):
        """
        Computes the homography to straighten the clueboard.
        """

        if len(good_matches) < 10:
            rospy.logwarn("Not enough good matches to compute Homography.")
            return

        ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        live_pts = np.float32([kp_live[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(ref_pts, live_pts, cv2.RANSAC, 5.0)

        if matrix is not None:
            x, y, w, h = cv2.boundingRect(blue_corners)
            h, w = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY).shape

            ref_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            live_corners = cv2.perspectiveTransform(ref_corners, matrix)

            # Ensure proper offset and shape
            offset = np.array([x, y], dtype=np.float32).reshape(1, 2)
            live_corners_full = live_corners.reshape(4, 2) + offset

            # Validate shapes before transformation
            if live_corners_full.shape != (4, 2) or live_corners_full.dtype != np.float32:
                rospy.logwarn("live_corners_full is not in the correct format.")
                return

            # Compute perspective transform
            straight_matrix = cv2.getPerspectiveTransform(
                live_corners_full,
                np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
            )
            # clueboard_transformed = cv2.warpPerspective(self.latest_frame, straight_matrix, (640, 480))

            # print("TRANSFORMED")
            # cv2.imshow("Clueboard", clueboard_transformed)
            # cv2.waitKey(1)

    def straighten_from_grey_clueboard(self, blue_roi):
        """
        Detects the edges of the grey clueboard within the blue ROI using Canny edge detection,
        finds the quadrilateral enclosing the grey clueboard, and applies a perspective transform.
        """
        if blue_roi is None:
            rospy.logwarn("No blue ROI provided.")
            return None

        # Convert blue_roi to grayscale
        gray_roi = cv2.cvtColor(blue_roi, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # Find contours from the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Identify the largest quadrilateral (assumed to be the grey clueboard)
        clueboard_corners = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Adjust this threshold as needed
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4 and area > max_area:
                    max_area = area
                    clueboard_corners = approx.reshape(4, 2)

        if clueboard_corners is None:
            # rospy.logwarn("No quadrilateral detected for grey clueboard.")
            return None

        # Sort the corners in a consistent order (top-left, top-right, bottom-right, bottom-left)
        def sort_corners(corners):
            s = corners.sum(axis=1)
            diff = np.diff(corners, axis=1)
            return np.array([
                corners[np.argmin(s)],  # Top-left
                corners[np.argmin(diff)],  # Top-right
                corners[np.argmax(s)],  # Bottom-right
                corners[np.argmax(diff)],  # Bottom-left
            ], dtype=np.float32)

        sorted_corners = sort_corners(clueboard_corners)

        # Define the target corners for the perspective transform (standard rectangle)
        width = 640  # Desired width
        height = 480  # Desired height
        dest_corners = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        # Compute the perspective transform matrix
        perspective_matrix = cv2.getPerspectiveTransform(sorted_corners, dest_corners)

        # Apply the perspective transform
        clueboard_transformed = cv2.warpPerspective(blue_roi, perspective_matrix, (width, height))

        # Debugging: Display the detected edges and the transformed clueboard
        # cv2.imshow("Detected Edges", edges)
        # cv2.imshow("Transformed Clueboard", clueboard_transformed)
        # cv2.waitKey(1)

        return clueboard_transformed


    def cnn(self,cropped_images):
        model = load_model("/home/fizzer/ros_ws/src/pink_chicken/node/trained_cnn_model_v2.h5")

        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        preprocessed_images = []
        for img in cropped_images:
            # Normalize pixel values to [0, 1]
            normalized = img / 255.0
            # Ensure 3 channels (convert grayscale to RGB if needed)
            if len(normalized.shape) == 2:  # Grayscale image
                uint8_image = (normalized * 255).astype('uint8')
                normalized = cv2.cvtColor(uint8_image, cv2.COLOR_GRAY2RGB)
            preprocessed_images.append(normalized)


        images = np.array(preprocessed_images)

        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        read_char = [characters[idx] for idx in predicted_classes]
        print("Recognized Characters:", " ".join(read_char))
        # model.summary()
        return read_char

def main(args):
    """
    Main entry point for the application.
    """
    # Initialize the Qt application

    app = QtWidgets.QApplication(args)

    # Create an instance of the read_sign class
    window = read_sign()
    window.show()  # Display the main window

    try:
        # Start the Qt application event loop
        app.exec_()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
        rospy.signal_shutdown("User closed the application")
    finally:
        # Ensure all OpenCV windows are closed
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    main(sys.argv)