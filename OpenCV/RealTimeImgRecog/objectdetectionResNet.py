import torch
import torchvision

import cv2
import requests
import numpy as np
import PIL


class ObjectDetectionResNet:

    def __init__(self, cam = "local", url=None):
        self.cam = cam
        if self.cam == "android": 
            self.url = url

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()

    def __call__(self):
        
        if self.cam == "local":
            player = self.get_video()
            assert player.isOpened() 
            x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
            y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))

            while(player.isOpened()):
                ret, frame = player.read()
                assert ret
                results = self.score_frame(frame)
                frame = self.plot_boxes(results,frame)
                cv2.imshow("video", frame)
                
                if cv2.waitKey(1) == 27 & 0xFF == ord('q'):
                    break
            player.release()
            
        else: 
            while True: 
                frame = self.get_android_cam(self.url)
                results = self.score_frame(frame)
                frame = self.plot_boxes(results,frame)
                frame = cv2.resize(frame, (852, 480))
                cv2.imshow("video", frame)
            
                if cv2.waitKey(1) == 27 & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

    def load_model(self):

        self.model_weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.model = torchvision.models.resnet50(weights=self.model_weights).to(self.device)

    def get_video(self):
        return cv2.VideoCapture(0)
    
    def get_android_cam(self, url):
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        return img
        
    def score_frame(self, frame): 
        self.model.to(self.device)
        self.model.eval()
        #PILframe = PIL.Image.fromarray(frame)
        with torch.inference_mode(): 
            #transformed_frame = self.transforms(PILframe).unsqueeze(dim=0)
            results = self.model(frame)
            labels, cord = results.xyxyn[0][:,-1].numpy(), results.xyxyn[0][:,:-1].numpy()
        
        return labels, cord

    def class_to_label(self, x): 
        return self.classes[int(x)]


    def plot_boxes(self, results, frame): 
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i] 
            if row[4] >= 0.2: 
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), \
                                int(row[2]*x_shape), int(row[3]*y_shape)

                bgr = (0,255,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), \
                        (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2 ,bgr, 2)

        return frame

