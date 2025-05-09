#!/usr/bin/env python2
import sys
import os


workspace_path = '/home/xuyuan/pan/volbox/devel/lib/python2.7/dist-packages'

if workspace_path not in sys.path:
    sys.path.append(workspace_path)
import rospy
import cv2
import numpy as np
import time
from sensor_msgs.msg import CameraInfo

import struct
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from mask_rcnn_ros.msg import Result
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import tf
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
class ColorBasedLocalization:
    def __init__(self):
        
        rospy.init_node('color_based_localization', anonymous=True)
        self.map_centroids = []
        self.depth_image = None
       
        self.colormap = None
        self.mask_rcnn_image = None
        self.robot_position = None

      
        self.bridge = CvBridge()
        self.camera_info_sub = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.camera_info_callback)
        self.depth_image_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_image_callback)
        self.mask_rcnn_sub = rospy.Subscriber("/mask_rcnn/visualization", Image, self.mask_rcnn_callback)
       
        self.pc_sub = rospy.Subscriber("/points_map", PointCloud2, self.colormap_callback)
     
        self.position_pub = rospy.Publisher('/initial_position', PoseStamped, queue_size=10)

       
        self.tf_listener = tf.TransformListener()
    def camera_info_callback(self, msg):
        self.camera_info = msg

    def mask_rcnn_callback(self, msg):   

        self.mask_rcnn_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
     
        lower_bound = np.array([0, 0, 200]) 
        upper_bound = np.array([100, 100, 255])
        centroids = []
        centroids_3d = []
        mask = cv2.inRange(self.mask_rcnn_image, lower_bound, upper_bound)
        moments = cv2.moments(mask)
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            centroids.append((cX, cY))  
        rospy.loginfo("Detected red object centroids: {}".format(centroids))
        
        if self.depth_image is not None and len(centroids) > 0:
      
            cX, cY = centroids[0]
            depth = self.depth_image[cY, cX] / 1000.0  
       
            X, Y, Z = self.image_to_3d(cX, cY, depth)
        
            rospy.loginfo("3D coordinates: ({}, {}, {})".format(X, Y, Z))
            centroids_3d.append((X,Y,Z)) 
        self.find_matching_location_in_map(centroids_3d)  


    def depth_image_callback(self, msg):
   
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    
    
        self.depth_image = depth_image  
      
        

    def get_camera_intrinsics(self):
   
        fx = self.camera_info.K[0]  
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]
        return fx, fy, cx, cy

    def image_to_3d(self, cX, cY, depth):
    
        fx, fy, cx, cy = self.get_camera_intrinsics()  

        X = (cX - cx) * depth / fx
        Y = (cY - cy) * depth / fy
        Z = depth  
    
        return X, Y, Z

     
    def colormap_callback(self, msg):  
       
  
        pc_data = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
    
        points = list(pc_data)
    
 
        xyz_points = np.array([(point[0], point[1], point[2]) for point in points])
        rgb_values = np.array([point[3] for point in points])
    
     
        rgb_int = np.array([struct.unpack('I', struct.pack('f', val))[0] for val in rgb_values])
        red_channel = (rgb_int & 0x00FF0000) >> 16
        green_channel = (rgb_int & 0x0000FF00) >> 8
        blue_channel = (rgb_int & 0x000000FF)
   
        red_threshold = 255
        green_threshold = 69
        blue_threshold = 20

  
        red_points = xyz_points[(red_channel == red_threshold) & (green_channel == green_threshold) & (blue_channel == blue_threshold)]
    
        clustering = DBSCAN(eps=0.1, min_samples=10).fit(red_points)


        labels = clustering.labels_


        unique_labels = set(labels)
        centroids = []

        for label in unique_labels:
            if label == -1:  
                continue
        
            cluster_points = red_points[labels == label]
    
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)

            print("Centroid of cluster {}: {}".format(label,centroid))


        print("Number of clusters (red objects): {}".format(len(centroids)))
        self.map_centroids = centroids
       

     
        
       
 
       

    def find_matching_location_in_map(self, centroids_3d):
        min_distance = float('inf')
        matching_cX, matching_cY, matching_cZ = None, None, None
        best_cX, best_cY, best_cZ = None, None, None
        if isinstance(centroids_3d, list) and len(centroids_3d) > 0:
       
            if len(centroids_3d) == 1:
                cX, cY, cZ = centroids_3d[0]
           
                for (map_cX, map_cY, map_cZ) in self.map_centroids:
                    distance = np.sqrt((cX - map_cX)**2 + (cY - map_cY)**2 + (cZ - map_cZ)**2)
                    if distance < min_distance:
                        min_distance = distance
                        matching_cX, matching_cY, matching_cZ = map_cX, map_cY, map_cZ
                        best_cX, best_cY, best_cZ = cX, cY, cZ
            else:
                for (cX, cY, cZ) in centroids_3d:
                    for (map_cX, map_cY, map_cZ) in self.map_centroids:
                        distance = np.sqrt((cX - map_cX)**2 + (cY - map_cY)**2 + (cZ - map_cZ)**2)
                        if distance < min_distance:
                            min_distance = distance
                            matching_cX, matching_cY, matching_cZ = map_cX, map_cY, map_cZ
                            best_cX, best_cY, best_cZ = cX, cY, cZ
   
        if matching_cX is not None and matching_cY is not None and matching_cZ is not None:
            rospy.loginfo("Matched centroid at: ({}, {}, {}) with distance: {:.2f}".format(matching_cX, matching_cY, matching_cZ, min_distance))

            centroids_3d_matched = [matching_cX, matching_cY, matching_cZ]
            best_3d = [best_cX, best_cY, best_cZ]
            transform_matrix = self.get_transform_matrix(centroids_3d_matched, best_3d)
            cX_final, cY_final, angle = self.apply_transform(transform_matrix)

            rospy.loginfo("Estimated robot position: ({}, {}, {})".format(cX_final, cY_final, angle))

        
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = 'map'
            pose.pose.position.x = cX_final
            pose.pose.position.y = cY_final
            pose.pose.position.z = 0

       
            quat = quaternion_from_euler(0, 0, angle)
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]

            self.position_pub.publish(pose)

        else:
            rospy.logwarn("No matching centroid found.")



        

        
        

        

    def get_transform_matrix(self, centroids_3d_matched, best_3d):

        centroids_3d_matched = np.array(centroids_3d_matched)
        best_3d = np.array(best_3d)
    
        if centroids_3d_matched.shape != (3,) or best_3d.shape != (3,):
            raise ValueError("centroids_3d_matched and best_3d should each be a 3D point (array of length 3).")
    

        translation = centroids_3d_matched - best_3d  
    
   

        map_angle = np.arctan2(centroids_3d_matched[1], centroids_3d_matched[0])
        detected_angle = np.arctan2(best_3d[1], best_3d[0])  
    
   
        rotation_angle = detected_angle - map_angle
    
        print("Translation: ", translation)
        print("Rotation angle (radians): ", rotation_angle)
    
        transform_matrix = {
        "translation": translation,
        "rotation": rotation_angle
        }
    
        return transform_matrix
        



    def apply_transform(self, transform_matrix):
        
        translation = transform_matrix["translation"]
        rotation_angle = transform_matrix["rotation"]
        
     
        cX_final = translation[0]
        cY_final = translation[1]
        
      
        angle = rotation_angle
        print(" apply_transform:{},{},{}".format(cX_final,cY_final,angle))
        return cX_final, cY_final, angle


    def transform_to_robot_base(self, map_x, map_y):
        try:
            
            (trans, rot) = self.tf_listener.lookupTransform('base_footprint', 'map', rospy.Time(0))

           
            map_point = np.array([map_x, map_y, 0, 1])  
            tf_matrix = tf.transformations.compose_matrix(translate=trans, angles=tf.transformations.euler_from_quaternion(rot))
            robot_point = np.dot(tf_matrix, map_point)

            rospy.loginfo("Transformed robot position: ({}, {})".format(robot_point[0],robot_point[1]))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF transform failed")

    def process(self):
        
        rospy.spin()

if __name__ == '__main__':
    localization_node = ColorBasedLocalization()
    localization_node.process()

