#!/usr/bin/env python2.7






import rospy
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image
import struct
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import tf
import tf2_ros
from sklearn.cluster import DBSCAN
class ColorBasedLocalization:
    def __init__(self):
      
        rospy.init_node('color_based_localization', anonymous=True)

    
        self.colormap = None
        self.mask_rcnn_image = None
        self.robot_position = None

 
        self.bridge = CvBridge()

       
        self.image_sub = rospy.Subscriber('/mask_rcnn/visualization', Image, self.mask_rcnn_callback)
        self.pc_sub = rospy.Subscriber("/points_map", PointCloud2, self.colormap_callback)

   
        self.position_pub = rospy.Publisher('/initial_position0', PoseStamped, queue_size=10)

     
        self.tf_listener = tf.TransformListener()

    def mask_rcnn_callback(self, msg):
    
        self.mask_rcnn_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
     
        lower_bound = np.array([0, 0, 200]) 
        upper_bound = np.array([100, 100, 255])

      
        mask = cv2.inRange(self.mask_rcnn_image, lower_bound, upper_bound)
        
   
        moments = cv2.moments(mask)
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])

           
            dx = 0.5  
            dy = 0.2  

          
            robot_x = cX + dx
            robot_y = cY + dy

        
            self.robot_position = (robot_x, robot_y)
            rospy.loginfo("Estimated robot position: ({}, {})".format(robot_x,robot_y))

            
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = 'map'  
            pose.pose.position.x = robot_x
            pose.pose.position.y = robot_y
            pose.pose.position.z = 0
            pose.pose.orientation.w = 1.0
            self.position_pub.publish(pose)

            
            self.find_matching_location_in_map()

    def colormap_callback0(self, msg):
       
        width = msg.info.width
        height = msg.info.height
        map_data = np.array(msg.data).reshape((height, width))

       
        self.colormap = map_data
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

       
        
        return centroids

    def find_matching_location_in_map(self):
       
        if self.colormap is None or self.mask_rcnn_image is None:
            return
        
       
        map_image = cv2.cvtColor(self.colormap, cv2.COLOR_GRAY2BGR)
        
        lower_bound = np.array([100, 0, 0])  
        upper_bound = np.array([255, 100, 100]) 

        map_mask = cv2.inRange(map_image, lower_bound, upper_bound)

        moments_map = cv2.moments(map_mask)
        if moments_map["m00"] != 0:
            cX_map = int(moments_map["m10"] / moments_map["m00"])
            cY_map = int(moments_map["m01"] / moments_map["m00"])

            rospy.loginfo("Matching location on the map: ({}, {})".format(cX_map, cY_map))

         
            self.transform_to_robot_base(cX_map, cY_map)

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

