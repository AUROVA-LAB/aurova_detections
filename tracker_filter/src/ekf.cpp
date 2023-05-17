#include "ekf.h"

CEkf::CEkf(ekf::KalmanConfiguration kalman_configuration)
{
  config_ = kalman_configuration;

  flag_ekf_initialised_ = false;
  debug_ = false;

   //State vector
  X_(0, 0) = 0.0;
  X_(1, 0) = 0.0;

  // State covariance matrix
  P_(0, 0) = 0.0; // initial value for x variance;
  P_(0, 1) = 0.0;

  P_(1, 0) = 0.0;
  P_(1, 1) = 0.0; // y variance

  // Model noise covariance matrix
  Q_(0, 0) = pow(config_.x_model, 2.0); //x noise variance
  Q_(0, 1) = 0.0;

  Q_(1, 0) = 0.0;
  Q_(1, 1) = pow(config_.y_model, 2.0); //y  noise variance


  // Identity matrix
  F_X_.setIdentity();
  F_q_.setIdentity();
  F_u_.setIdentity();
  H_.setIdentity();
}

CEkf::~CEkf(void)
{

}

//Instead of static prediction, the prediction trends to go to the origin, so the robot stops slowly, as there isn't localization.
void CEkf::predict(double t, double search_time, Eigen::Matrix<double, 2, 1>& last_state)
{
  X_=last_state*(1-pow((t/search_time),2));

  if (flag_ekf_initialised_)
  {
    // Covariance prediction
    P_ = F_X_ * P_ * F_X_.transpose() + F_q_ * Q_ * F_q_.transpose();
  }
}

bool CEkf::update(ekf::Observation obs)
{

  if (!flag_ekf_initialised_)
  {
    X_(0) = obs.x;
    X_(1) = obs.y;
    P_ = obs.sigma; // initial value for covariance;
    flag_ekf_initialised_=true;
    return true;
  }
  else
  {
    //Filling the observation vector
    Eigen::Matrix<double, 2, 1> y;
    y(0) = obs.x;
    y(1) = obs.y;

    // Expectation
    Eigen::Matrix<double, 2, 1> e, z; // expectation, innovation
    e(0) = X_(0);
    e(1) = X_(1);

    // Innovation
    z = y - e;

    // Innovation covariance
    Eigen::Matrix<double, 2, 2> R = Eigen::Matrix<double, 2, 2>::Zero();

    R = obs.sigma;

    Eigen::Matrix<double, 2, 2> Z = Eigen::Matrix<double, 2, 2>::Zero();

    Z = H_ * P_ * H_.transpose() + R;

    if (debug_)
      std::cout << "CEkf::Update R: " << R << std::endl;

    double mahalanobis_distance = sqrt(z.transpose() * Z.inverse() * z);

    if (debug_)
      std::cout << "CEkf::Update mahalanobis_distance: " << mahalanobis_distance << std::endl;

    if (mahalanobis_distance < config_.outlier_mahalanobis_threshold)
    {
      // Kalman gain
      Eigen::Matrix<double, 2, 2> K = Eigen::Matrix<double, 2, 2>::Zero();
      K = P_ * H_.transpose() * Z.inverse();

      if (debug_)
        std::cout << "CEkf::Update K: " << K << std::endl;

      // State correction
      X_ = X_ + K * z;

      if (debug_)
        std::cout << "CEkf::Update X_: " << X_ << std::endl;

      // State covariance correction
      //P_ = P_ - K * Z * K.transpose();
      // State covariance correction (Joseph form)
      Eigen::Matrix<double, 2, 2> I = Eigen::Matrix<double, 2, 2>::Identity();
      P_ = (I - K * H_) * P_;

      if (debug_)
        std::cout << "CEkf::Update P_: " << P_ << std::endl;
        return true;
    }
    return false;
  }
  
}

void CEkf::getStateAndCovariance(Eigen::Matrix<double, 2, 1> &state, Eigen::Matrix<double, 2, 2> &covariance)
{
  state = X_;
  covariance = P_;
}

void CEkf::setStateAndCovariance(Eigen::Matrix<double, 2, 1> state, Eigen::Matrix<double, 2, 2> covariance)
{
  X_ = state;
  P_ = covariance;
}

void CEkf::set_config(ekf::KalmanConfiguration config){
  config_=config;
  // Model noise covariance matrix
  Q_(0, 0) = pow(config_.x_model, 2.0); //x noise variance
  Q_(0, 1) = 0.0;

  Q_(1, 0) = 0.0;
  Q_(1, 1) = pow(config_.y_model, 2.0); //y  noise variance
}