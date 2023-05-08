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
  this->private_node_handle_.getParam("metrics", metrics);
  
  this->private_node_handle_.getParam("x_model", config_.x_model);
  this->private_node_handle_.getParam("y_model", config_.y_model);
  ekf::KalmanConfiguration ekf_config; ekf_config.outlier_mahalanobis_threshold=config_.max_traslation;
  ekf_config.x_model=config_.x_model; ekf_config.y_model=config_.y_model;

  if(metrics){
    string label_name, metrics_name;
    this->private_node_handle_.getParam("label_file", label_name);
    this->private_node_handle_.getParam("output_file", metrics_name);
    metrics_file.open(metrics_name,ios::trunc);
    ifstream label_file(label_name, ios::in);
    string s;labelData data;
    while(label_file>>s){
      data.time=stod(s); label_file>>s;
      data.ix=stoi(s); label_file>>s;
      data.iy=stoi(s); label_file>>s;
      data.ex=stoi(s); label_file>>s;
      data.ey=stoi(s);
      ground_truth.push_back(data);
    }
    ground_truth_id=0;
    last_ground_truth(0)=0; last_ground_truth(1)=0;
    this->ground_truth_ekf = new CEkf(ekf_config);
  }

  this->ekf = new CEkf(ekf_config);
  // this->ekf->setDebug(true);
  flag_rate=true; flag_tracking=false;
  range_sub.subscribe(this->private_node_handle_, "/ouster/range_image",  10);
  yolo_sub.subscribe(this->private_node_handle_, "/tracker/yolo" , 10);
  dasiam_sub.subscribe(this->private_node_handle_, "/tracker/dasiamrpn" , 10);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, detection_msgs::BoundingBoxes, detection_msgs::BoundingBoxes> MySyncPolicy;
  static message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), range_sub, yolo_sub, dasiam_sub);
  sync.registerCallback(boost::bind(&TrackerFilterAlgNode::callback,this, _1, _2, _3));
  
  pc_filtered_pub = this->private_node_handle_.advertise<PointCloud> ("/ouster_filtered", 1);  
  goal_pub = this->private_node_handle_.advertise<geometry_msgs::PoseWithCovarianceStamped>( "/target", 1 );
  gt_pub = this->private_node_handle_.advertise<geometry_msgs::PoseWithCovarianceStamped>( "/ground_truth", 1 );
  searchBB_pub = this->private_node_handle_.advertise<detection_msgs::BoundingBoxes>( "/tracker_filter/search_area", 1 );

  pointCloud_msg = PointCloud::Ptr (new PointCloud);
  time_pointcloud=0; time_update=0;
}

TrackerFilterAlgNode::~TrackerFilterAlgNode(void)
{
  // [free dynamic memory]
  if(metrics_file.is_open()) metrics_file.close();
  cout<<"Pointcloud reconstruction mean time "<<time_pointcloud*pow(10,-9)<<endl;
  cout<<"Update mean time "<<time_update*pow(10,-9)<<endl;
}

void TrackerFilterAlgNode::mainNodeThread(void)
{
  //lock access to algorithm if necessary
  this->alg_.lock();
  // [fill msg structures]
  
  // [fill srv structure and make request to the server]
  
  // [fill action structure and make request to the action server]
  flag_rate=true;
  //Predict covariance
  if(ekf->flag_ekf_initialised_){
    double t=ros::Time::now().toSec()-last_detection;
    //Check for prevent errors in case rosbag fails (only simulations)
    if(t<0){
      last_detection=ros::Time::now().toSec();
      t=config_.search_time;
    }
    t=min(t,config_.search_time);
    ekf->predict(t, config_.search_time, last_state);
  } 
  if(metrics) ground_truth_ekf->predict(0.0, config_.search_time, last_ground_truth);
  // [publish messages]
  //Construct the pointcloud
  if(flag_image){
    flag_image=false;
    auto start=chrono::high_resolution_clock::now();
    PointCloud::Ptr cloud (new PointCloud);
    pointCloud_msg->clear(); // clear data pointcloud
    Eigen::Matrix<double, 2, 1> state; Eigen::Matrix<double, 2, 2> covariance;
    ekf->getStateAndCovariance(state,covariance);
    for (uint iy = 0;iy<im_rows; iy++){
      float ang_h = 22.5 - (45.0/128.0)*iy;
      ang_h = ang_h*M_PI/180.0;
      Eigen::VectorXf Z_row = data_metrics.row(iy)*sin(ang_h);

      for (uint ix = 0;ix<im_cols; ix++){        
        // Point cloud reconstruction
        if (data_metrics(iy,ix)==0)
          continue;
        
        float ang_w = 184.0 - (360.0/2048.0)*ix;
        ang_w = ang_w*M_PI/180.0;
        
        float z = Z_row(ix);
        float aux = sqrt(pow(data_metrics(iy,ix),2)-pow(z,2));
        float y = aux*sin(ang_w);
        float x = aux*cos(ang_w);

        pcl::PointXYZ point(x,y,z);
        //Remove points of the target.
        if (sqrt(pow(x-state(0),2)+pow(y-state(1),2))>config_.filter_radious){          
          cloud->push_back(point); 
        }
      }
    }
    pointCloud_msg->points=cloud->points;
    pointCloud_msg->is_dense = true;
    pointCloud_msg->width = (int) pointCloud_msg->points.size();
    pointCloud_msg->height = 1;
    pointCloud_msg->header.frame_id = "os_sensor";
    ros::Time time_st = ros::Time::now(); // Para PCL se debe modificar el stamp y no se puede usar directamente el del topic de entrada
    pointCloud_msg->header.stamp = time_st.toNSec()/1e3;
    pc_filtered_pub.publish (pointCloud_msg);
    double diff = chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now()-start).count();
    time_pointcloud=(time_pointcloud+diff)/2.0;
  }
  
  this->alg_.unlock();
}

Eigen::Vector2d TrackerFilterAlgNode::boundingBox2point(detection_msgs::BoundingBox& bb, cv::Mat& img_range){

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
  auto start=chrono::high_resolution_clock::now();
  flag_rate=false; flag_image=true;
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
  im_rows=img_range.rows; im_cols=img_range.cols;
  Eigen::Matrix<float,Dynamic,Dynamic> depth_data;// matrix with image values and matrix qith image values into real range data
  cv2eigen(img_range,depth_data);       // convert img_range into eigen matrix
  data_metrics = depth_data*(261/pow(2,16)); // resolution 16 bits -> 4mm. 
  bool target_detected=false;
  

  bool new_obs=false;
  uint num_dasiam_detection = dasiam_msg->bounding_boxes.size();
  detection_msgs::BoundingBox ground_truth_bb;
  if(metrics){
    double t=yolo_msg->header.stamp.toSec();
    
    //Get actual bounding box
    while (t>ground_truth[ground_truth_id+1].time){
      ground_truth_id++;
      if (ground_truth_id+1==ground_truth.size()){
        metrics=false; break;
      }
    }
    if(metrics){
      float w0=1-(t-ground_truth[ground_truth_id].time)/(ground_truth[ground_truth_id+1].time-ground_truth[ground_truth_id].time);
      float w1=1-(ground_truth[ground_truth_id+1].time-t)/(ground_truth[ground_truth_id+1].time-ground_truth[ground_truth_id].time);
      ground_truth_bb.xmin=int(ground_truth[ground_truth_id].ix*w0+ground_truth[ground_truth_id+1].ix*w1);
      ground_truth_bb.ymin=int(ground_truth[ground_truth_id].iy*w0+ground_truth[ground_truth_id+1].iy*w1);
      ground_truth_bb.xmax=int(ground_truth[ground_truth_id].ex*w0+ground_truth[ground_truth_id+1].ex*w1);
      ground_truth_bb.ymax=int(ground_truth[ground_truth_id].ey*w0+ground_truth[ground_truth_id+1].ey*w1);
      metrics_file<<std::fixed<<t<<" ";
    }
  } 
  if (num_dasiam_detection>0){
    //If there are 2 boxes, then the target is between the boundaries of the image.
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
    Eigen::Vector2d point=boundingBox2point(bb,img_range);
    ekf::Observation obs; obs.x=point(0); obs.y=point(1);
    obs.sigma_x=dasiam_msg->bounding_boxes[0].probability; obs.sigma_y=obs.sigma_x;
    new_obs = ekf->update(obs);
    if(metrics){
      //Distance between the center of the bounding boxes.
      float cx1=ground_truth_bb.xmin+(ground_truth_bb.xmax-ground_truth_bb.xmin)/2.0;
      float cy1=ground_truth_bb.ymin+(ground_truth_bb.ymax-ground_truth_bb.ymin)/2.0;
      float cx2=bb.xmin+(bb.xmax-bb.xmin)/2.0;
      float cy2=bb.ymin+(bb.ymax-bb.ymin)/2.0;
      metrics_file<<float(sqrt(pow(cx1-cx2,2)+pow(cy1-cy2,2)))<<" ";
      //Intersection over Union
      metrics_file<<get_iou(ground_truth_bb,bb)<<" ";
    }
  }
  else if(metrics) metrics_file<<"None None ";

  if (yolo_msg->bounding_boxes.size()>0){
    detection_msgs::BoundingBox bb=yolo_msg->bounding_boxes[0];
    Eigen::Vector2d point=boundingBox2point(bb,img_range);
    ekf::Observation obs; obs.x=point(0); obs.y=point(1);
    obs.sigma_x=yolo_msg->bounding_boxes[0].probability; obs.sigma_y=obs.sigma_x;
    bool ok = ekf->update(obs);
    new_obs = new_obs || ok;
    if(metrics){
      //Distance between the center of the bounding boxes.
      float cx1=ground_truth_bb.xmin+(ground_truth_bb.xmax-ground_truth_bb.xmin)/2.0;
      float cy1=ground_truth_bb.ymin+(ground_truth_bb.ymax-ground_truth_bb.ymin)/2.0;
      float cx2=bb.xmin+(bb.xmax-bb.xmin)/2.0;
      float cy2=bb.ymin+(bb.ymax-bb.ymin)/2.0;
      metrics_file<<float(sqrt(pow(cx1-cx2,2)+pow(cy1-cy2,2)))<<" ";
      //Intersection over Union
      metrics_file<<get_iou(ground_truth_bb,bb)<<" ";
    }
  }
  else if(metrics) metrics_file<<"None None ";

  Eigen::Matrix<double, 2, 1> state; Eigen::Matrix<double, 2, 2> covariance;
  ekf->getStateAndCovariance(state,covariance);

  if(metrics){
    Eigen::Vector2d ground_truth_pose=boundingBox2point(ground_truth_bb,img_range);
    //Check if the position have sense respect the previous position, that  is the target depth is OK.
    ekf::Observation obs; obs.x=ground_truth_pose(0); obs.y=ground_truth_pose(1);
    obs.sigma_x=0.1; obs.sigma_y=0.1;
    ground_truth_ekf->update(obs);
    Eigen::Matrix<double, 2, 2> aux;
    ground_truth_ekf->getStateAndCovariance(ground_truth_pose,aux);
    last_ground_truth=ground_truth_pose;
    metrics_file<<sqrt(pow(ground_truth_pose(0)-state(0),2)+pow(ground_truth_pose(1)-state(1),2))<<endl;
  }

  detection_msgs::BoundingBox search_bbox;
  if(!flag_tracking){
    search_bbox.ymin=0; search_bbox.ymax=img_range.rows;
    search_bbox.xmin=0; search_bbox.xmax=img_range.cols;
  }
  else{
    //Calculate search bbox. The search area is the one that have been filtered, in the image plane.
    
    search_bbox.ymin=0; search_bbox.ymax=img_range.rows;
    float ang_w=atan2(state(1),state(0));
    float ang_wmin=atan2(state(1)+(config_.filter_radious+covariance(1,1))*cos(ang_w),state(0)-(config_.filter_radious+covariance(0,0))*sin(ang_w));
    ang_wmin = ang_wmin*180.0/M_PI;
    search_bbox.xmin = int((184 - ang_wmin)*(2048.0/360.0));

    float ang_wmax=atan2(state(1)-(config_.filter_radious+covariance(1,1))*cos(ang_w),state(0)+(config_.filter_radious+covariance(0,0))*sin(ang_w));
    ang_wmax = ang_wmax*180.0/M_PI;
    search_bbox.xmax = int((184 - ang_wmax)*(2048.0/360.0));
    if(search_bbox.xmin>search_bbox.xmax) search_bbox.xmin-=img_range.cols;
    //We use the 'probability' field to pass the covariance.
    search_bbox.probability=(covariance(0,0)+covariance(1,1))/2;
  }

   if(new_obs){
    last_detection=ros::Time::now().toSec(); flag_tracking=true;
    last_state=state;
  }
  else if(sqrt(state(0)*state(0)+state(1)*state(1))<0.01){
    flag_tracking=false; ekf->flag_ekf_initialised_=false;
  }
  auto time_st=ros::Time::now();
  detection_msgs::BoundingBoxes search_msg; search_msg.header=dasiam_msg->header;
  search_msg.header.stamp=time_st; search_msg.bounding_boxes.push_back(search_bbox);
  searchBB_pub.publish(search_msg);

  geometry_msgs::PoseWithCovarianceStamped goal_msg;
  goal_msg.header.stamp=time_st; goal_msg.header.frame_id= "os_sensor";
  goal_msg.pose.pose.position.x=state(0); goal_msg.pose.pose.position.y=state(1);
  goal_msg.pose.covariance[0]=covariance(0,0); goal_msg.pose.covariance[7]=covariance(1,1);
  goal_msg.pose.covariance[14]=1;  
  goal_pub.publish(goal_msg);

  if(metrics){
    geometry_msgs::PoseWithCovarianceStamped gt_msg;
    gt_msg.header.stamp=time_st; gt_msg.header.frame_id= "os_sensor";
    gt_msg.pose.pose.position.x=last_ground_truth(0); gt_msg.pose.pose.position.y=last_ground_truth(1);
    gt_pub.publish(gt_msg);
  }
  double diff = chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now()-start).count();
  time_update=(time_update+diff)/2.0;
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
  if(config.max_traslation!=this->config_.max_traslation ||
    config.x_model!=config_.x_model || config.y_model!=config_.y_model){
    ekf::KalmanConfiguration ekf_config; ekf_config.outlier_mahalanobis_threshold=config.max_traslation;
    ekf_config.x_model=config.x_model; ekf_config.y_model=config.y_model;
    ekf->set_config(ekf_config);
  }
  this->config_=config;
  this->alg_.unlock();
}

void TrackerFilterAlgNode::addNodeDiagnostics(void)
{
}

float TrackerFilterAlgNode::get_iou(detection_msgs::BoundingBox bb1, detection_msgs::BoundingBox bb2){

    int x_left = max(bb1.xmin, bb2.xmin);
    int y_top = max(bb1.ymin, bb2.ymin);
    int x_right = min(bb1.xmax, bb2.xmax);
    int y_bottom = min(bb1.ymax, bb2.ymax);

    if (x_right < x_left || y_bottom < y_top)
        return 0.0;

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);

    float bb1_area = (bb1.xmax - bb1.xmin) * (bb1.ymax - bb1.ymin);
    float bb2_area = (bb2.xmax - bb2.xmin) * (bb2.ymax - bb2.ymin);

    float iou = intersection_area / float(bb1_area + bb2_area - intersection_area);
    return iou;
}

/* main function */
int main(int argc,char *argv[])
{
  return algorithm_base::main<TrackerFilterAlgNode>(argc, argv, "tracker_filter_alg_node");
}
