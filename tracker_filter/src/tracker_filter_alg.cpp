#include "tracker_filter_alg.h"

TrackerFilterAlgorithm::TrackerFilterAlgorithm(void)
{
  pthread_mutex_init(&this->access_,NULL);
}

TrackerFilterAlgorithm::~TrackerFilterAlgorithm(void)
{
  pthread_mutex_destroy(&this->access_);
}

void TrackerFilterAlgorithm::config_update(Config& config, uint32_t level)
{
  this->lock();

  // save the current configuration
  this->config_=config;
  
  this->unlock();
}

// TrackerFilterAlgorithm Public API
