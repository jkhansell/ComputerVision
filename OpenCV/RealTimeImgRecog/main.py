import objectdetectionyolo as od 


cameradetect = od.ObjectDetectionYolo(cam="android", \
                    url="http://172.26.43.61:8080/shot.jpg")
cameradetect()


