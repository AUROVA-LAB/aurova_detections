#include "tracker_filter_alg_node.h"

using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

TrackerFilterAlgNode::TrackerFilterAlgNode(void) :
  algorithm_base::IriBaseAlgorithm<TrackerFilterAlgorithm>()
{
  //init class attributes if necessary
  if(!this->private_node_handle_.getParam("rate", this->config_.rate))
  {
    ROS_WARN("TrackerFilterAlgNode::TrackerFilterAlgNode: param 'rate' not found");
  }
  else
    this->setRate(this->config_.rate);

  this->private_node_handle_.getParam("filtering_method", filt_method);
  this->private_node_handle_.getParam("bounding_box_percet_reduction", config_.bb_red);
  this->private_node_handle_.getParam("max_traslation_distance", config_.max_traslation);
  this->private_node_handle_.getParam("filter_radious", config_.filter_radious);
  this->private_node_handle_.getParam("search_time", config_.search_time);
  this->private_node_handle_.getParam("front_image_cov1", config_.fr_im_cov1);
  this->private_node_handle_.getParam("front_image_cov2", config_.fr_im_cov2);


  // this->ekf = new CEkf(config_.max_traslation);
  // this->ekf->setDebug(true);
  prev_state = Eigen::Vector2d::Zero();
  flag_rate=true; flag_tracking=false;
  range_sub.subscribe(this->private_node_handle_, "/ouster/range_image",  10);
  yolo_sub.subscribe(this->private_node_handle_, "/tracker/yolo" , 10);
  dasiam_sub.subscribe(this->private_node_handle_, "/tracker/dasiamrpn" , 10);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, detection_msgs::BoundingBoxes, detection_msgs::BoundingBoxes> MySyncPolicy;
  static message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), range_sub, yolo_sub, dasiam_sub);
  sync.registerCallback(boost::bind(&TrackerFilterAlgNode::callback,this, _1, _2, _3));
  
  pc_filtered_pub = this->private_node_handle_.advertise<PointCloud> ("/ouster_filtered", 1);  
  goal_pub = this->private_node_handle_.advertise<geometry_msgs::PoseWithCovarianceStamped>( "/target", 1 );
  searchBB_pub = this->private_node_handle_.advertise<detection_msgs::BoundingBoxes>( "/tracker_filter/search_area", 1 );

  pointCloud_msg = PointCloud::Ptr (new PointCloud);
}

TrackerFilterAlgNode::~TrackerFilterAlgNode(void)
{
  // [free dynamic memory]
}

void TrackerFilterAlgNode::mainNodeThread(void)
{
  //lock access to algorithm if necessary
  this->alg_.lock();
  // [fill msg structures]
  
  // [fill srv structure and make request to the server]
  
  // [fill action structure and make request to the action server]
  flag_rate=true;
  // [publish messages]
  
  this->alg_.unlock();
}

Eigen::Vector2d TrackerFilterAlgNode::boundingBox2point(detection_msgs::BoundingBox& bb, cv::Mat& img_range, 
                                              const Eigen::Ref<const Eigen::MatrixXf>& data_metrics){

  float median = 0;
  // Calculate the median depth

  float depth_ave = 0;   //  average distance of object
  uint cont_pix=0;        // number of pixels 

  uint start_x = (1-config_.bb_red/2.0) * bb.xmin + (config_.bb_red/2.0 * bb.xmax);
  uint end_x   = (1-config_.bb_red/2.0) * bb.xmax + (config_.bb_red/2.0 * bb.xmin);
  uint start_y = (1-config_.bb_red/2.0) * bb.ymin + (config_.bb_red/2.0 * bb.ymax);
  uint end_y   = (1-config_.bb_red/2.0) * bb.ymax + (config_.bb_red/2.0 * bb.ymin);
  // optimizar recortando matrix de direccion a hasta b y luego sacar media
  std::vector<float> vec_std_depth; // vector para medianas

  for (int iy = start_y;iy<end_y; iy++)
    for (int ix = start_x;ix<end_x; ix++)
        if(data_metrics(iy,remap(ix,img_range.cols))!=0){
          depth_ave += data_metrics(iy,remap(ix,img_range.cols));
          cont_pix++;
          vec_std_depth.push_back(data_metrics(iy,remap(ix,img_range.cols)));
        }
    // condicion para que no de valores infinitos     
  if(depth_ave == 0 && cont_pix==0){
    cont_pix = 1;
    vec_std_depth.push_back(0);
  }

  // punto medio 
  int Ox = (bb.xmax+bb.xmin)/2;
  int Oy = (bb.ymax+bb.ymin)/2;
  float p_med = data_metrics(Oy,remap(Ox,img_range.cols));

  /////// calculo de la mediana ////////////////////////////////////////
  int n = sizeof(vec_std_depth) / sizeof(vec_std_depth[0]);  
  sort(vec_std_depth.begin(), vec_std_depth.begin() + n, greater<int>());
  int tam = vec_std_depth.size();
  
  if (tam % 2 == 0) {  
      median = (vec_std_depth[((tam)/2) -1] + vec_std_depth[(tam)/2])/2.0; 
  }      
  else { 
      if(tam==1)
      median = vec_std_depth[tam];
    else
      median = vec_std_depth[tam/2];
  }         
  vec_std_depth.clear();

  //Calculate goal as the  center of the bounding box but using the median depth
  Eigen::Vector2d obs;
  float ang_h = 22.5 - (45.0/128.0)*Oy;
  ang_h = ang_h*M_PI/180.0;
  float ang_w = 184.0 - (360.0/2048.0)*remap(Ox,img_range.cols);
  ang_w = ang_w*M_PI/180.0;
  float z = median * sin(ang_h);
  obs(1) = sqrt(pow(median,2)-pow(z,2))*sin(ang_w);
  obs(0) = sqrt(pow(median,2)-pow(z,2))*cos(ang_w);
  
  return obs;
}

/*  [subscriber callbacks] */

void TrackerFilterAlgNode::callback(const ImageConstPtr& in_image,const boost::shared_ptr<const detection_msgs::BoundingBoxes>& yolo_msg,
                                    const boost::shared_ptr<const detection_msgs::BoundingBoxes>& dasiam_msg)
{
  if(!flag_rate) return;
  flag_rate=false;
  pointCloud_msg->clear(); // clear data pointcloud
  cv_bridge::CvImagePtr cv_range;
      try
      {
        cv_range = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::MONO16);
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }

  cv::Mat img_range  = cv_range->image; // get image matrix of cv_range
  Eigen::Matrix<float,Dynamic,Dynamic> depth_data , data_metrics;// matrix with image values and matrix qith image values into real range data
  cv2eigen(img_range,depth_data);       // convert img_range into eigen matrix
  data_metrics = depth_data*(261/pow(2,16)); // resolution 16 bits -> 4mm. 
  bool target_detected=false;
  PointCloud::Ptr point_cloud (new PointCloud);
  PointCloud::Ptr cloud (new PointCloud);

  point_cloud->width = img_range.cols; 
  point_cloud->height = img_range.rows;
  point_cloud->is_dense = false;
  point_cloud->points.resize (point_cloud->width * point_cloud->height);
  uint num_pix = 0;

  uint num_dasiam_detection = dasiam_msg->bounding_boxes.size();

  vector<Eigen::Vector2d> observations(0);
  Eigen::Vector2d state = Eigen::Vector2d::Zero();
  if (num_dasiam_detection>0){
    //If there are 2 boxes, then the target is backwards, inj the limits of the image.
    detection_msgs::BoundingBox bb;
    if (num_dasiam_detection==2){
      bb.ymin = dasiam_msg->bounding_boxes[0].ymin;
      bb.ymax = dasiam_msg->bounding_boxes[0].ymax;
      if(dasiam_msg->bounding_boxes[0].xmin==0)
        bb.xmax=dasiam_msg->bounding_boxes[0].xmax, bb.xmin=dasiam_msg->bounding_boxes[1].xmin-img_range.cols;
      else
        bb.xmax=dasiam_msg->bounding_boxes[1].xmax, bb.xmin=dasiam_msg->bounding_boxes[0].xmin-img_range.cols;
    }
    else if (num_dasiam_detection==1){
      bb.ymin = dasiam_msg->bounding_boxes[0].ymin;
      bb.ymax = dasiam_msg->bounding_boxes[0].ymax;
      bb.xmin = dasiam_msg->bounding_boxes[0].xmin;
      bb.xmax = dasiam_msg->bounding_boxes[0].xmax;
    }
    observations.push_back(boundingBox2point(bb,img_range,data_metrics));
  }

  if (yolo_msg->bounding_boxes.size()>0){
    detection_msgs::BoundingBox bb=yolo_msg->bounding_boxes[0];
    observations.push_back(boundingBox2point(bb,img_range,data_metrics));
  }
  
  int count=0;
  for(auto obs:observations){
    Eigen::Vector2d diff=obs-prev_state;
    //Reject outliers
    if(!flag_tracking || sqrt(diff(0)*diff(0)+diff(1)*diff(1))<config_.max_traslation){
      state+=obs; count++;
    }
  }

  if(count>0){
    state=state/count;
    //depth_ave = depth_ave/cont_pix;
    // std::cout<<"Depth average: "<<depth_ave << " median: "<<median << "Punto medio: "<<p_med<< std::endl;  
    // std::cout<<"Goal x: "<<goal_x<<" Y: "<<goal_y<<" Z: "<<z<<std::endl;
    for (uint iy = 0;iy<img_range.rows; iy++){
      for (uint ix = 0;ix<img_range.cols; ix++){        

        // recosntruccion de la nube de puntos
        if (data_metrics(iy,ix)==0)
          continue;

        float ang_h = 22.5 - (45.0/128.0)*iy;
        ang_h = ang_h*M_PI/180.0;
        float ang_w = 184.0 - (360.0/2048.0)*ix;
        ang_w = ang_w*M_PI/180.0;

        float z = data_metrics(iy,ix) * sin(ang_h);
        float y = sqrt(pow(data_metrics(iy,ix),2)-pow(z,2))*sin(ang_w);
        float x = sqrt(pow(data_metrics(iy,ix),2)-pow(z,2))*cos(ang_w);

        point_cloud->points[num_pix].x = x;
        point_cloud->points[num_pix].y = y;
        point_cloud->points[num_pix].z = z;
        //Remove points of the target.
        if (sqrt(pow(x-state(0),2)+pow(y-state(1),2))>config_.filter_radious){          
          cloud->push_back(point_cloud->points[num_pix]); 
        }
        num_pix++; 
      }
    }
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *pointCloud_msg, indices);
    last_detection=ros::Time::now().toSec(); flag_tracking=true;
    prev_state=state;
  }
  
  //Track the target in the point cloud.
  else{
    //The target is lost, and as there isn't localization, the robot have to stop slowly.
    double t=ros::Time::now().toSec()-last_detection;
    //Check for prevent errors in case rosbag fails (only simulations)
    t=min(t,config_.search_time);
    state=prev_state*(1-pow((t/config_.search_time),3));

    for (uint iy = 0;iy<img_range.rows; iy++){
      for (uint ix = 0;ix<img_range.cols; ix++){        

        // recosntruccion de la nube de puntos
        if (data_metrics(iy,ix)==0)
          continue;

        float ang_h = 22.5 - (45.0/128.0)*iy;
        ang_h = ang_h*M_PI/180.0;
        float ang_w = 184.0 - (360.0/2048.0)*ix;
        ang_w = ang_w*M_PI/180.0;

        float z = data_metrics(iy,ix) * sin(ang_h);
        float y = sqrt(pow(data_metrics(iy,ix),2)-pow(z,2))*sin(ang_w);
        float x = sqrt(pow(data_metrics(iy,ix),2)-pow(z,2))*cos(ang_w);
        point_cloud->points[num_pix].x = x;
        point_cloud->points[num_pix].y = y;
        point_cloud->points[num_pix].z = z;
        if(sqrt(pow(x-state(0),2)+pow(y-state(1),2))>config_.filter_radious){
          cloud->push_back(point_cloud->points[num_pix]);
        }
        num_pix++;
      }
    }
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *pointCloud_msg, indices);
    if(sqrt(state(0)*state(0)+state(1)-state(1))<0.01) flag_tracking=false;
  }

  pointCloud_msg->is_dense = true;
  pointCloud_msg->width = (int) pointCloud_msg->points.size();
  pointCloud_msg->height = 1;
  pointCloud_msg->header.frame_id = "os_sensor";
  ros::Time time_st = ros::Time::now(); // Para PCL se debe modificar el stamp y no se puede usar directamente el del topic de entrada
  pointCloud_msg->header.stamp = time_st.toNSec()/1e3;
  pc_filtered_pub.publish (pointCloud_msg);

  geometry_msgs::PoseWithCovarianceStamped goal_msg;
  goal_msg.header.stamp=time_st; goal_msg.header.frame_id= "os_sensor";
  goal_msg.pose.pose.position.x=state(0); goal_msg.pose.pose.position.y=state(1);
  goal_pub.publish(goal_msg);

  detection_msgs::BoundingBox search_bbox;
  if(!flag_tracking){
    search_bbox.ymin=0; search_bbox.ymax=img_range.rows;
    search_bbox.xmin=0; search_bbox.xmax=img_range.cols;
  }
  else{
    //Calculate search bbox. The search area is the one that have been filtered, in the image plane.
    
    search_bbox.ymin=0; search_bbox.ymax=img_range.rows;
    float ang_w=atan2(state(1),state(0));

    float ang_wmin=atan2(state(1)+config_.max_traslation*cos(ang_w),state(0)-config_.filter_radious*sin(ang_w));
    ang_wmin = ang_wmin*180.0/M_PI;
    search_bbox.xmin = int((180 - ang_wmin)*(2048.0/360.0));

    float ang_wmax=atan2(state(1)-config_.max_traslation*cos(ang_w),state(0)+config_.filter_radious*sin(ang_w));
    ang_wmax = ang_wmax*180.0/M_PI;
    search_bbox.xmax = int((180 - ang_wmax)*(2048.0/360.0));
    if(search_bbox.xmin>search_bbox.xmax) search_bbox.xmin-=img_range.cols;
  }
  detection_msgs::BoundingBoxes search_msg; search_msg.header=dasiam_msg->header;
  search_msg.header.stamp=time_st; search_msg.bounding_boxes.push_back(search_bbox);
  searchBB_pub.publish(search_msg);
}

int TrackerFilterAlgNode::remap(int x, int limit){
  if(x<0) return limit+x;
  return x;
}

/*  [service callbacks] */

/*  [action callbacks] */

/*  [action requests] */

void TrackerFilterAlgNode::node_config_update(Config &config, uint32_t level)
{
  this->alg_.lock();
  if(config.rate!=this->getRate())
    this->setRate(config.rate);
  this->config_=config;
  this->alg_.unlock();
}

void TrackerFilterAlgNode::addNodeDiagnostics(void)
{
}

/* main function */
int main(int argc,char *argv[])
{
  return algorithm_base::main<TrackerFilterAlgNode>(argc, argv, "tracker_filter_alg_node");
}