#!/usr/bin/env python2
import sys
import os

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

from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import tf
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler




import time

class ColorBasedLocalization:
    def __init__(self):
        
        self.depth_image = None
        self.camera_info = None 
        self.bridge = CvBridge()
        
        
        self.camera_info_sub = rospy.Subscriber('/camera/info', CameraInfo, self.camera_info_callback)
       
        self.depth_image_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_image_callback)
        
        self.mask_rcnn_sub = rospy.Subscriber('/mask_rcnn/visualization', Image, self.mask_rcnn_callback)
        
        self.pc_sub = rospy.Subscriber("/points_map", PointCloud2, self.colormap_callback)
        self.position_pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size=10)

    def camera_info_callback(self, msg):
      
        self.camera_info = msg
        rospy.loginfo("Received camera info")

    def depth_image_callback(self, msg):
       
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        rospy.loginfo("Received depth image")

    def mask_rcnn_callback(self, msg):
        rospy.loginfo("mask_rcnn_callback triggered")   
        self.mask_rcnn_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
       
        lower_bound = np.array([0, 0, 200])  
        upper_bound = np.array([100, 100, 255])
        
        centroids = []
        centroids_2d = []
        
        mask = cv2.inRange(self.mask_rcnn_image, lower_bound, upper_bound)
        moments = cv2.moments(mask)
        
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            centroids.append((cX, cY))

        if self.depth_image is not None and len(centroids) > 0:
            
            cX, cY = centroids[0]
            depth = self.depth_image[cY, cX] / 1000.0  
            
            
            X, Y, Z = self.image_to_3d(cX, cY, depth)
            centroids_2d = [X,Y,Z]
            rospy.loginfo("3D coordinates: ({}, {}, {})".format(X, Y, Z))
        
        self.find_matching_location_in_map(centroids_2d)  

    def image_to_3d(self, cX, cY, depth):
       
        fx, fy, cx, cy = self.get_camera_intrinsics() 
        
        
        X = (cX - cx) * depth / fx
        Y = (cY - cy) * depth / fy
        Z = depth  
        
        return X, Y, Z

    def get_camera_intrinsics(self):
        
        if self.camera_info is not None:
            fx = self.camera_info.K[0]  
            fy = self.camera_info.K[4]
            cx = self.camera_info.K[2]
            cy = self.camera_info.K[5]
            return fx, fy, cx, cy
        else:
            rospy.logwarn("Camera info not received yet")
            return None, None, None, None
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

    def find_matching_location_in_map(self, centroids):
        if len(centroids) < 2:
            rospy.loginfo("Detected positions too few, waiting for more data...")
            return
        min_distance = float('inf')
        matching_cX, matching_cY, matching_cZ = None, None, None
        centroids_3d = []
    
        for (cX, cY, cZ) in centroids:
            for (map_cX, map_cY, map_cZ) in self.map_centroids:
   
                distance = np.sqrt((cX - map_cX)**2 + (cY - map_cY)**2 + (cZ - map_cZ)**2)
            
            
                if distance < min_distance:
                    min_distance = distance
                    matching_cX, matching_cY, matching_cZ = cX, cY, cZ
        centroids_3d = [cX, cY, cZ]
        transform_matrix = self.get_transform_matrix(centroids_3d)

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

    def get_transform_matrix(self, centroids):
       
        
        centroids_2d = np.array(self.map_centroids)[:, :2]  # Keep only the first two columns (x, y)
        translation = np.mean(centroids_2d, axis=0) - np.mean(centroids, axis=0)


        
      
        map_angle = np.arctan2(self.map_centroids[1][1] - self.map_centroids[0][1], 
                               self.map_centroids[1][0] - self.map_centroids[0][0])

        detected_angle = np.arctan2(centroids[1][1] - centroids[0][1], 
                                    centroids[1][0] - centroids[0][0])

      
        rotation_angle = detected_angle - map_angle
        print(" get_transform_matrix:{},{}".format(translation,rotation_angle))
       
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

