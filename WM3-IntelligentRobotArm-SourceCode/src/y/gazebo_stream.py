import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from ultralytics import YOLO

class ObjectRecognitionNode(Node):
    def __init__(self):
        super().__init__('object_recognition_node')
        self.subscription = self.create_subscription(
            Image,
            '/pi_camera/image_raw',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()

        # Paths for input and output images
        self.input_image_path = '/home/eireland/ty/src/y/gazebo_img/captured_image.jpg'

        # Load YOLO model
        self.model = YOLO("/home/eireland/ty/src/y/best.pt")

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Save the captured image, overwriting each time
        cv2.imwrite(self.input_image_path, cv_image)
        self.get_logger().info(f"Image saved to {self.input_image_path}")

        # Run object detection on the saved image and save results in the output directory
        self.run_object_detection()

    def run_object_detection(self):
        # Use the YOLO model to predict and save results
        # The save_dir must be a directory, not a file path
        self.model(self.input_image_path, save=True)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
