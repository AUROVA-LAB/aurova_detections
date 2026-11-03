#include "tracker_filter_alg_node.h"

using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

TrackerFilterAlgNode::TrackerFilterAlgNode(void) :
  Node("trackerfilter")
{
  //init class attributes if necessary
  this->declare_parameter("bounding_box_percet_reduction", 0.6);
  this->get_parameter("bounding_box_percet_reduction", this->bb_red);
  this->declare_parameter("max_translation_distance", 2.0);
  this->get_parameter("max_translation_distance", this->max_translation);
  this->declare_parameter("filter_radious", 1.5);
  this->get_parameter("filter_radious", this->filter_radious);
  this->declare_parameter("search_time", 2.0);
  this->get_parameter("search_time", this->search_time);
  this->declare_parameter("metrics",false);
  this->get_parameter("metrics", metrics);
  
  this->declare_parameter("x_model", 0.2);
  this->get_parameter("x_model", this->x_model);
  this->declare_parameter("y_model", 0.2);
  this->get_parameter("y_model", this->y_model);
  this->declare_parameter("theta_w", 5.0);
  this->get_parameter("theta_w", this->theta_w);
  this->declare_parameter("mute_image_trackers", false);
  this->get_parameter("mute_image_trackers", this->mute_image_trackers);
  this->declare_parameter("pc_frame", "os_sensor");
  this->get_parameter("pc_frame", this->pc_frame);
  ekf::KalmanConfiguration ekf_config; ekf_config.outlier_mahalanobis_threshold=this->max_translation;
  ekf_config.x_model=this->x_model; ekf_config.y_model=this->y_model;

  if(metrics){
    string label_name, metrics_name;
    this->declare_parameter("label_file", "");
    this->get_parameter("label_file", label_name);
    this->declare_parameter("output_file", "");
    this->get_parameter("output_file", metrics_name);
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
  flag_rate=false; flag_tracking=false; m2track_obs=false; flag_image=false;
  range_sub.subscribe(this, "/ouster/range_image");
  yolo_sub.subscribe(this, "/tracker/yolo");
  dasiam_sub.subscribe(this, "/tracker/dasiamrpn");
  this->sync = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(MySyncPolicy(10),range_sub, yolo_sub,dasiam_sub);
  this->sync->registerCallback(std::bind(&TrackerFilterAlgNode::callback,this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  m2track_subscriber = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/m2track", 1, std::bind(&TrackerFilterAlgNode::m2track_callback, this, std::placeholders::_1));
  
  pc_filtered_pub = this->create_publisher<sensor_msgs::msg::PointCloud2> ("/ouster_filtered", 1);  
  goal_pub = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>( "/target", 1 );
  gt_pub = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>( "/ground_truth", 1 );
  searchBB_pub = this->create_publisher<detection_msgs::msg::BoundingBoxes>( "/tracker_filter/search_area", 1 );

  pointCloud_msg = PointCloud::Ptr (new PointCloud);
  time_pointcloud=0; time_update=0;
  auto parameter_callback = this->add_on_set_parameters_callback(std::bind(&TrackerFilterAlgNode::node_config_update, this, std::placeholders::_1));
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

  // [fill msg structures]
  
  // [fill srv structure and make request to the server]
  
  // [fill action structure and make request to the action server]
  flag_rate=true;

  //Predict covariance
  if(ekf->flag_ekf_initialised_){
    double t=this->now().seconds()-last_detection;
    //Check for prevent errors in case rosbag fails (only simulations)
    if(t<0){
      last_detection=this->now().seconds();
      t=this->search_time;
    }
    t=min(t,this->search_time);
    ekf->predict(t, this->search_time, last_state);
  } 
  if(metrics) ground_truth_ekf->predict(0.0, this->search_time, last_ground_truth);
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
        if (sqrt(pow(x-state(0),2)+pow(y-state(1),2))>this->filter_radious){          
          cloud->push_back(point); 
        }
      }
    }
    pointCloud_msg->points=cloud->points;
    pointCloud_msg->is_dense = true;
    pointCloud_msg->width = (int) pointCloud_msg->points.size();
    pointCloud_msg->height = 1;
    this->publishPointCloud(pc_filtered_pub, pointCloud_msg, this->pc_frame, this->msg_time);
    double diff = chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now()-start).count();
    time_pointcloud=(time_pointcloud+diff)/2.0;
  }
}

Eigen::Vector2d TrackerFilterAlgNode::boundingBox2point(detection_msgs::msg::BoundingBox& bb, cv::Mat& img_range){

  float median = 0;
  // Calculate the median depth

  float depth_ave = 0;   //  average distance of object
  uint cont_pix=0;        // number of pixels 

  uint start_x = (1-this->bb_red/2.0) * bb.xmin + (this->bb_red/2.0 * bb.xmax);
  uint end_x   = (1-this->bb_red/2.0) * bb.xmax + (this->bb_red/2.0 * bb.xmin);
  uint start_y = (1-this->bb_red/2.0) * bb.ymin + (this->bb_red/2.0 * bb.ymax);
  uint end_y   = (1-this->bb_red/2.0) * bb.ymax + (this->bb_red/2.0 * bb.ymin);
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

void TrackerFilterAlgNode::callback(const sensor_msgs::msg::Image::ConstSharedPtr&  in_image, 
          const detection_msgs::msg::BoundingBoxes::ConstSharedPtr&  yolo_msg, const detection_msgs::msg::BoundingBoxes::ConstSharedPtr& dasiam_msg)
{
  if(!flag_rate) return;
  this->msg_time = in_image->header.stamp;
  auto start=chrono::high_resolution_clock::now();
  flag_rate=false; 
  geometry_msgs::msg::PoseWithCovarianceStamped m2track_copy=m2track_msg;
  cv_bridge::CvImagePtr cv_range;
  try
  {
    cv_range = cv_bridge::toCvCopy(in_image, sensor_msgs::image_encodings::MONO16);
  }
  catch (cv_bridge::Exception& e)
  {
    RCLCPP_ERROR(this->get_logger(),"cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat img_range  = cv_range->image; // get image matrix of cv_range
  im_rows=img_range.rows; im_cols=img_range.cols;
  Eigen::Matrix<float,Dynamic,Dynamic> depth_data;// matrix with image values and matrix qith image values into real range data
  cv2eigen(img_range,depth_data);       // convert img_range into eigen matrix
  data_metrics = depth_data*(261/pow(2,16)); // resolution 16 bits -> 4mm. 
  flag_image=true;
  

  bool new_obs=false;
  uint num_dasiam_detection = dasiam_msg->bounding_boxes.size();
  detection_msgs::msg::BoundingBox ground_truth_bb;
  if(metrics){
    double t=yolo_msg->header.stamp.sec + 1e-9*yolo_msg->header.stamp.nanosec;
    
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

  if(m2track_obs){
    m2track_obs=false;
    ekf::Observation obs; obs.x=m2track_copy.pose.pose.position.x; obs.y=m2track_copy.pose.pose.position.y;
    obs.sigma=Eigen::Matrix2d::Zero();
    obs.sigma(0,0)=m2track_copy.pose.covariance[0]; obs.sigma(1,1)=m2track_copy.pose.covariance[7];
    new_obs = ekf->update(obs);
  }

  Eigen::Matrix<double, 2, 1> state; Eigen::Matrix<double, 2, 2> covariance;
  ekf->getStateAndCovariance(state,covariance);

  float dist_yolo=-1, iou_yolo=-1; //In previous version, dasiam metrics were written before.
  if (yolo_msg->bounding_boxes.size()>0 && !(this->mute_image_trackers && flag_tracking)){
    detection_msgs::msg::BoundingBox bb=yolo_msg->bounding_boxes[0];
    Eigen::Vector2d point=boundingBox2point(bb,img_range);
    ekf::Observation obs; obs.x=point(0); obs.y=point(1);
    //Covariance is greater in the angle of detection, because the depth estimation may have errors.
    Eigen::Matrix2d cov=Eigen::Matrix2d::Zero(), rot=Eigen::Matrix2d::Zero();
    rot=Eigen::Rotation2Dd(atan2(obs.y,obs.x));
    
    float cov_value = yolo_msg->bounding_boxes[0].probability;
    // cov_value+=pow(obs.x-state(0),2)+pow(obs.y-state(1),2); //Increase covariance if the detection is far from the prediction.
    cov(0,0)=cov_value*2; cov(1,1)=cov_value;
    obs.sigma=rot*cov*rot.transpose();
    bool ok = ekf->update(obs);
    new_obs = new_obs || ok;
    if(metrics){
      //Distance between the center of the bounding boxes.
      float cx1=ground_truth_bb.xmin+(ground_truth_bb.xmax-ground_truth_bb.xmin)/2.0;
      float cy1=ground_truth_bb.ymin+(ground_truth_bb.ymax-ground_truth_bb.ymin)/2.0;
      float cx2=bb.xmin+(bb.xmax-bb.xmin)/2.0;
      float cy2=bb.ymin+(bb.ymax-bb.ymin)/2.0;
      dist_yolo=float(sqrt(pow(cx1-cx2,2)+pow(cy1-cy2,2)));
      iou_yolo=get_iou(ground_truth_bb,bb);
    }
  }

  if (num_dasiam_detection>0 && !(this->mute_image_trackers && flag_tracking)){
    //If there are 2 boxes, then the target is between the boundaries of the image.
    detection_msgs::msg::BoundingBox bb;
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
    Eigen::Matrix2d cov=Eigen::Matrix2d::Zero(), rot=Eigen::Matrix2d::Zero();
    rot=Eigen::Rotation2Dd(atan2(obs.y,obs.x));
    float cov_value = dasiam_msg->bounding_boxes[0].probability;
    // cov_value+=(pow(obs.x-state(0),2)+pow(obs.y-state(1),2)); 
    cov(0,0)=cov_value*2; cov(1,1)=cov_value;
    obs.sigma=rot*cov*rot.transpose();
    bool ok = ekf->update(obs);
    new_obs = ok || new_obs;
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

  if(metrics){
    if(dist_yolo<0) metrics_file<<"None None ";
    else metrics_file<<dist_yolo<<" "<<iou_yolo<<" ";
  }

  ekf->getStateAndCovariance(state,covariance);

  if(metrics){
    Eigen::Vector2d ground_truth_pose=boundingBox2point(ground_truth_bb,img_range);
    //Check if the position have sense respect the previous position, that  is the target depth is OK.
    ekf::Observation obs; obs.x=ground_truth_pose(0); obs.y=ground_truth_pose(1);
    obs.sigma=Eigen::Matrix2d::Zero();
    obs.sigma(0,0)=0.1; obs.sigma(1,1)=0.1;
    ground_truth_ekf->update(obs);
    Eigen::Matrix<double, 2, 2> aux;
    ground_truth_ekf->getStateAndCovariance(ground_truth_pose,aux);
    last_ground_truth=ground_truth_pose;
    metrics_file<<sqrt(pow(ground_truth_pose(0)-state(0),2)+pow(ground_truth_pose(1)-state(1),2))<<endl;
  }

  detection_msgs::msg::BoundingBox search_bbox;
  if(!flag_tracking){
    search_bbox.ymin=0; search_bbox.ymax=img_range.rows;
    search_bbox.xmin=0; search_bbox.xmax=img_range.cols;
  }
  else{
    //Calculate search bbox. The search area is the one that have been filtered, in the image plane.
    
    search_bbox.ymin=0; search_bbox.ymax=img_range.rows;
    float ang_w=atan2(state(1),state(0));
    // float ang_wmin=atan2(state(1)+(this->filter_radious*1.5+covariance(1,1))*cos(ang_w),state(0)-(this->filter_radious*1.5+covariance(0,0))*sin(ang_w));
    Eigen::Vector2d aux,normal;
    normal(0)=sin(ang_w); normal(1)=-cos(ang_w);
    aux = normal/sqrt(normal.transpose()*covariance.inverse()*normal);
    float ang_wmin=atan2(state(1)+aux(1),state(0)+aux(0));
    ang_wmin = ang_wmin*180.0/M_PI -this->theta_w;
    search_bbox.xmax = int((184 - ang_wmin)*(2048.0/360.0));

    // float ang_wmax=atan2(state(1)-(this->filter_radious*1.5+covariance(1,1))*cos(ang_w),state(0)+(this->filter_radious*1.5+covariance(0,0))*sin(ang_w));
    float ang_wmax=atan2(state(1)-aux(1),state(0)-aux(0));
    ang_wmax = ang_wmax*180.0/M_PI+this->theta_w;
    search_bbox.xmin = int((184 - ang_wmax)*(2048.0/360.0));
    if(search_bbox.xmin>search_bbox.xmax) search_bbox.xmin-=img_range.cols;
    //We use the 'probability' field to pass the covariance.
    search_bbox.probability=(covariance(0,0)+covariance(1,1))/2;
  }

  if(new_obs){
    last_detection=this->now().seconds(); flag_tracking=true;
    last_state=state;
  }
  else if(sqrt(state(0)*state(0)+state(1)*state(1))<0.01){
    flag_tracking=false; ekf->flag_ekf_initialised_=false;
  }
  auto time_st=this->now();
  detection_msgs::msg::BoundingBoxes search_msg; search_msg.header=dasiam_msg->header;
  search_msg.header.stamp=time_st; search_msg.bounding_boxes.push_back(search_bbox);
  searchBB_pub->publish(search_msg);

  geometry_msgs::msg::PoseWithCovarianceStamped goal_msg;
  goal_msg.header.stamp=time_st; goal_msg.header.frame_id= this->pc_frame;
  goal_msg.pose.pose.position.x=state(0); goal_msg.pose.pose.position.y=state(1);
  goal_msg.pose.pose.orientation.w=1.0; 
  goal_msg.pose.covariance[0]=covariance(0,0); goal_msg.pose.covariance[7]=covariance(1,1);
  goal_msg.pose.covariance[1]=covariance(0,1); goal_msg.pose.covariance[6]=covariance(1,0);
  goal_msg.pose.covariance[14]=1;  
  goal_pub->publish(goal_msg);

  if(metrics){
    geometry_msgs::msg::PoseWithCovarianceStamped gt_msg;
    gt_msg.header.stamp=time_st; gt_msg.header.frame_id= this->pc_frame;
    gt_msg.pose.pose.position.x=last_ground_truth(0); gt_msg.pose.pose.position.y=last_ground_truth(1);
    gt_pub->publish(gt_msg);
  }
  double diff = chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now()-start).count();
  time_update=(time_update+diff)/2.0;
}

void TrackerFilterAlgNode::m2track_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr  data){
  m2track_msg=*data;
  m2track_obs=true;
}

int TrackerFilterAlgNode::remap(int x, int limit){
  if(x<0) return limit+x;
  return x;
}
/*  [service callbacks] */

/*  [action callbacks] */

/*  [action requests] */

rcl_interfaces::msg::SetParametersResult TrackerFilterAlgNode::node_config_update(const std::vector<rclcpp::Parameter> &parameters)
{

  double rate;
  this->get_parameter("rate", rate);
  if(rate!=this->rate)
  {
    this->rate = rate;
    this->loop_rate = std::make_shared<rclcpp::Rate>(this->rate);
  }
  this->get_parameter("bounding_box_percet_reduction", this->bb_red);
  this->get_parameter("max_translation_distance", this->max_translation);
  this->get_parameter("filter_radious", this->filter_radious);
  this->get_parameter("search_time", this->search_time);
  this->get_parameter("x_model", this->x_model);
  this->get_parameter("y_model", this->y_model);
  this->get_parameter("theta_w", this->theta_w);
  ekf::KalmanConfiguration ekf_config; ekf_config.outlier_mahalanobis_threshold=this->max_translation;
  ekf_config.x_model=this->x_model; ekf_config.y_model=this->y_model;
  this->ekf->set_config(ekf_config);
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";
  return result;

}


float TrackerFilterAlgNode::get_iou(detection_msgs::msg::BoundingBox bb1, detection_msgs::msg::BoundingBox bb2){

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

void TrackerFilterAlgNode::publishPointCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher, 
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud, std::string frame, rclcpp::Time now)
{
    sensor_msgs::msg::PointCloud2::SharedPtr pc2_cloud(new sensor_msgs::msg::PointCloud2);

    pcl::toROSMsg(*point_cloud, *pc2_cloud);
    pc2_cloud->header.frame_id = frame;
    pc2_cloud->header.stamp = now;
    publisher->publish(*pc2_cloud);
}


/* main function */
int main(int argc,char *argv[])
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<TrackerFilterAlgNode>();
	node->rate = node->declare_parameter<double>("rate", 10.0);
	node->loop_rate = std::make_shared<rclcpp::Rate>(node->rate);

	while (rclcpp::ok()) {
		node->mainNodeThread();
		rclcpp::spin_some(node);
		node->loop_rate->sleep();
	}
	rclcpp::shutdown();
	return 0;
}