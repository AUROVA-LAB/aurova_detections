# ROS Point Cloud Tracker

This package uses a Deep Neural Network to track a target in the point cloud. Especifically, it uses the M2-Track from [Open3DSOT](https://github.com/Ghostish/Open3DSOT), but it can work with others models. For using other models, adapt the [inference module](src/inference/__init__.py). Because of the different versions of software for Machine Learning, it's recommended to use the docker image available in this repository. See [Docker](https://docs.docker.com/) for more information.



