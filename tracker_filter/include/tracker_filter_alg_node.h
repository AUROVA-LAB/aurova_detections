// Copyright (C) 2010-2011 Institut de Robotica i Informatica Industrial, CSIC-UPC.
// Author 
// All rights reserved.
//
// This file is part of iri-ros-pkg
// iri-ros-pkg is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// 
// IMPORTANT NOTE: This code has been generated through a script from the 
// iri_ros_scripts. Please do NOT delete any comments to guarantee the correctness
// of the scripts. ROS topics can be easly add by using those scripts. Please
// refer to the IRI wiki page for more information:
// http://wikiri.upc.es/index.php/Robotics_Lab

#ifndef _tracker_filter_alg_node_h_
#define _tracker_filter_alg_node_h_

#include <iri_base_algorithm/iri_base_algorithm.h>
#include "tracker_filter_alg.h"
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <image_transport/image_transport.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_spherical.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/registration/icp.h>

#include <iostream>
#include <fstream>
#include <math.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <detection_msgs/BoundingBox.h>
#include <detection_msgs/BoundingBoxes.h>
#include "ekf.h"

#include <pcl/filters/statistical_outlier_removal.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <limits>
#include <chrono> 

// [publisher subscriber headers]

// [service client headers]

// [action server client headers]

/**
 * \brief IRI ROS Specific Algorithm Class
 *
 */

struct labelData
{
  double time;
  int ix,iy,ex,ey;
};

class TrackerFilterAlgNode : public algorithm_base::IriBaseAlgorithm<TrackerFilterAlgorithm>
{
  private:
    typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
    ros::Publisher pc_filtered_pub; // publisher de la imagen de puntos filtrada
    ros::Publisher goal_pub, gt_pub; // Goal pose
    ros::Publisher searchBB_pub; // Search bounding box
    PointCloud::Ptr pointCloud_msg;
    message_filters::Subscriber<sensor_msgs::Image>range_sub;
    message_filters::Subscriber<detection_msgs::BoundingBoxes> yolo_sub, dasiam_sub;   
    std::string filt_method= "median";
    // CEkfPtr ekf;

    //Variables
    bool flag_rate, flag_tracking, flag_image, metrics;
    std::ofstream metrics_file;
    std::vector<labelData> ground_truth; u_int ground_truth_id; 
    double last_detection;
    Eigen::Matrix<double, 2, 1> last_state, last_ground_truth;
    CEkfPtr ekf, ground_truth_ekf;
    Eigen::MatrixXf data_metrics;
    uint im_rows,im_cols;

   /**
    * \brief config variable
    *
    * This variable has all the driver parameters defined in the cfg config file.
    * Is updated everytime function config_update() is called.
    */
    Config config_;
  public:
   /**
    * \brief Constructor
    * 
    * This constructor initializes specific class attributes and all ROS
    * communications variables to enable message exchange.
    */
    TrackerFilterAlgNode(void);

   /**
    * \brief Destructor
    * 
    * This destructor frees all necessary dynamic memory allocated within this
    * this class.
    */
    ~TrackerFilterAlgNode(void);

    void callback(const sensor_msgs::ImageConstPtr& in_image, const detection_msgs::BoundingBoxesConstPtr& yolo,const boost::shared_ptr<const detection_msgs::BoundingBoxes>& dasiam);
    int remap(int x, int limit);
    float get_iou(detection_msgs::BoundingBox bb1, detection_msgs::BoundingBox bb2);

    Eigen::Vector2d boundingBox2point(detection_msgs::BoundingBox& bb, cv::Mat& im_range);

  protected:
   /**
    * \brief main node thread
    *
    * This is the main thread node function. Code written here will be executed
    * in every node loop while the algorithm is on running state. Loop frequency 
    * can be tuned by modifying loop_rate attribute.
    *
    * Here data related to the process loop or to ROS topics (mainly data structs
    * related to the MSG and SRV files) must be updated. ROS publisher objects 
    * must publish their data in this process. ROS client servers may also
    * request data to the corresponding server topics.
    */
    void mainNodeThread(void);

   /**
    * \brief dynamic reconfigure server callback
    * 
    * This method is called whenever a new configuration is received through
    * the dynamic reconfigure. The derivated generic algorithm class must 
    * implement it.
    *
    * \param config an object with new configuration from all algorithm 
    *               parameters defined in the config file.
    * \param level  integer referring the level in which the configuration
    *               has been changed.
    */
    void node_config_update(Config &config, uint32_t level);

   /**
    * \brief node add diagnostics
    *
    * In this abstract function additional ROS diagnostics applied to the 
    * specific algorithms may be added.
    */
    void addNodeDiagnostics(void);

    // [diagnostic functions]
    
    // [test functions]
};

#endif
