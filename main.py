import torch
import cv2
import numpy as np
from pathlib import Path
import yaml
from PIL import Image

class WeedGuard:
    def __init__(self, weights_path='weights/best.pt', conf_threshold=0.25):
        
        self.conf_threshold = conf_threshold
        
        
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                  path=weights_path, force_reload=True)
        self.model.conf = conf_threshold
        
        
        self.class_names = ['weed']  
        
    def preprocess_image(self, image):
       
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
       
        return image_rgb

    def detect_weeds(self, image):
        
        processed_img = self.preprocess_image(image)
        
        results = self.model(processed_img)
        
        detections = []
        for *box, conf, cls in results.xyxy[0]:  # xyxy format
            if conf >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, box)
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class': self.class_names[int(cls)]
                }
                detections.append(detection)
        
        annotated_image = image.copy()
        for det in detections:
            bbox = det['bbox']
            cv2.rectangle(annotated_image, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]),
                         (0, 255, 0), 2)
            
            
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(annotated_image, label,
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 0), 2)
        
        return detections, annotated_image

    def process_video_stream(self, source=0):
        
        cap = cv2.VideoCapture(source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            
            detections, annotated_frame = self.detect_weeds(frame)
            
            cv2.imshow('WeedGuard Detection', annotated_frame)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

class WeedDatasetPreparation:
    """Helper class for preparing weed detection dataset"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'
        
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
    def create_data_yaml(self, train_path, val_path):
     
        data_yaml = {
            'train': str(train_path),
            'val': str(val_path),
            'nc': 1,  
            'names': ['weed']  
        }
        
        with open(self.data_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)
            
    def convert_to_yolo_format(self, bbox, img_width, img_height):
      
        x1, y1, x2, y2 = bbox
        
      
        x_center = (x1 + x2) / (2 * img_width)
        y_center = (y1 + y2) / (2 * img_height)
        
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return [x_center, y_center, width, height]

def train_model(data_yaml_path, epochs=100, batch_size=16, img_size=640):
   
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=img_size
    )

if __name__ == "__main__":
    
    weed_detector = WeedGuard(weights_path='weights/best.pt')
    
    image = cv2.imread('test_image.jpg')
    detections, annotated_image = weed_detector.detect_weeds(image)
    
    cv2.imshow('WeedGuard Detection', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
