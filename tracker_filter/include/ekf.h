#ifndef _ekf_h_
#define _ekf_h_

#include <iostream>
#include <Eigen/Dense>
#include "ros/ros.h"
#include "math.h"

#define PI 3.14159265358979323846

namespace ekf
{

struct Observation
{
  double x, y;
  double sigma_x, sigma_y;
};

struct KalmanConfiguration
{
  double x_model, y_model;
  double outlier_mahalanobis_threshold;
};
}

class CEkf;
typedef CEkf* CEkfPtr;

class CEkf
{
private:
  Eigen::Matrix<double, 2, 1> X_;
  Eigen::Matrix<double, 2, 2> F_X_;
  Eigen::Matrix<double, 2, 2> F_u_;
  Eigen::Matrix<double, 2, 2> F_q_;
  Eigen::Matrix<double, 2, 2> Q_;
  Eigen::Matrix<double, 2, 2> P_;
  Eigen::Matrix<double, 2, 2> H_;

  ekf::KalmanConfiguration config_;

  double wheelbase_;

  bool debug_;

public:

  bool flag_ekf_initialised_;

  CEkf(ekf::KalmanConfiguration kalman_configuration);

  ~CEkf(void);

  void predict(double t, double search_time,Eigen::Matrix<double, 2, 1>& last_state);

  bool update(ekf::Observation obs);

  void getStateAndCovariance(Eigen::Matrix<double, 2, 1>& state, Eigen::Matrix<double, 2, 2>& covariance);
  
  void setStateAndCovariance(Eigen::Matrix<double, 2, 1> state, Eigen::Matrix<double, 2, 2> covariance);

  void setDebug(bool debug)
  {
    debug_ = debug;
  }

  void set_config(ekf::KalmanConfiguration config);

};

#endif
