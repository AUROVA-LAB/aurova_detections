PASOS EJECUCION

1. Modificar en yolov5.launch los parametros del sistema. Algunos interesantes son:
	-weights
	-data
	-confidence_threshold
	-iou_threshold
	-inference_size_h
	-inference_size_w
	-input_image_topic
	-output_topic
	-publish_image
	-output_image_topic

2. Ir con la terminal a la ruta donde este el Dockerfile y compilar la docker con:
	sudo docker build -t yolov5_ouster .

3. Una vez compilada, ejecutar la docker con: 
	sudo docker run --shm-size=1g --gpus all --cpuset-cpus="0-1" --ulimit memlock=-1 --ulimit stack=67108864 --rm -it --name yolov5_ouster --network host -v ~/:/blue-onboard yolov5_ouster

4. Dentro de la docker se ejecuta automaticamente detect.py y merged_channels_ouster, por lo que no hace falta nada mas. El programa se queda esperando a recibir el topic imagenes del ouster (signal, reflec, nearir, range)

5. Se publican los topics de deteccion de personas como: 


	sensor_msgs/Image            ---> /ouster_merged
	sensor_msgs/Image            ---> /yolov5/image_out
	detection_msgs/BoundingBoxes ---> /yolov5/detections
