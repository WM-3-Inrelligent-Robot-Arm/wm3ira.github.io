import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import os
import cv2
from ultralytics import YOLO
import numpy as np
import tf2_ros
import tf2_geometry_msgs

class ObjectCoordinateNode(Node):
    def __init__(self):
        super().__init__('object_coordinate_node')

        # Directory where images are saved
        self.image_dir = '/home/eireland/ty/runs/obb/predict'
        self.last_processed_image = None

        # Load the YOLO model
        self.model = YOLO("/home/eireland/ty/src/y/best.pt")

        # Create a TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher for the object's PoseStamped message
        self.pose_publisher = self.create_publisher(PoseStamped, 'object_pose', 10)

        # Camera Intrinsics (example values, should be obtained from your camera calibration)
        self.fx = 600  # Focal length in x (in pixels)
        self.fy = 600  # Focal length in y (in pixels)
        self.cx = 320  # Optical center x (image center)
        self.cy = 240  # Optical center y (image center)

        # Depth assumption (can be replaced with actual depth data)
        self.depth = 1.0  # in meters

        # Timer to periodically check for new images
        self.timer = self.create_timer(1.0, self.process_latest_image)  # Check every second

    def process_latest_image(self):
        # List all images in the directory
        images = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        if not images:
            self.get_logger().info("No images found in the directory.")
            return

        # Sort images by modification time to get the latest one
        images.sort(key=lambda x: os.path.getmtime(os.path.join(self.image_dir, x)))
        latest_image = images[-1]

        # Check if this image has already been processed
        if latest_image == self.last_processed_image:
            return

        self.last_processed_image = latest_image
        image_path = os.path.join(self.image_dir, latest_image)
        self.get_logger().info(f"Processing image: {image_path}")

        # Read the image using OpenCV
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            self.get_logger().error(f"Failed to load image: {image_path}")
            return

        # Run object detection using YOLO
        results = self.model(cv_image)

        # Check if any objects were detected
        if len(results) > 0:
            detections = results[0].boxes  # Get the bounding boxes of the detected objects
            if detections is not None:
                # Process the first detected object (for simplicity)
                for box in detections.xywh:
                    center_x, center_y, width, height = box[0], box[1], box[2], box[3]

                    # Calculate the real-world coordinates of the object (x, y, z)
                    object_position = self.pixel_to_real(center_x, center_y, self.depth)
                    self.get_logger().info(f"Object Position (in camera frame): {object_position}")

                    # Create a PoseStamped message to send to the robot
                    object_pose = PoseStamped()
                    object_pose.header.stamp = self.get_clock().now().to_msg()
                    object_pose.header.frame_id = "camera_link"  # Camera's frame of reference
                    object_pose.pose.position.x = object_position[0]
                    object_pose.pose.position.y = object_position[1]
                    object_pose.pose.position.z = object_position[2]

                    # Publish the PoseStamped message
                    self.pose_publisher.publish(object_pose)
                    self.get_logger().info(f"Published object pose: {object_pose}")

                    # Transform object position to robot's base frame
                    try:
                        transform = self.tf_buffer.lookup_transform("base_link", "camera_link", rclpy.time.Time())
                        transformed_pose = tf2_geometry_msgs.do_transform_pose(object_pose, transform)
                        self.get_logger().info(f"Transformed Object Pose (in robot frame): {transformed_pose.pose}")
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                        self.get_logger().error(f"TF error: {e}")

    def pixel_to_real(self, center_x, center_y, depth):
        """
        Converts pixel coordinates (center_x, center_y) of the bounding box
        to real-world coordinates (x, y, z) in the camera frame.
        :param center_x: Center x coordinate of the bounding box
        :param center_y: Center y coordinate of the bounding box
        :param depth: Depth (distance from the camera to the object in meters)
        :return: [x, y, z] coordinates in meters in the camera frame
        """
        x = (center_x - self.cx) * depth / self.fx
        y = (center_y - self.cy) * depth / self.fy
        z = depth  # Assuming depth is directly from the camera sensor

        return [x, y, z]

def main(args=None):
    rclpy.init(args=args)
    node = ObjectCoordinateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
