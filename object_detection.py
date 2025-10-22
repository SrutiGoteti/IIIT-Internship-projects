from ultralytics import YOLO

#Loads a pretrained model YOLOv8 for object detection
model = YOLO('yolov8n.pt')

#Test the model with an online image and store the info in a list
results = model('https://ultralytics.com/images/zidane.jpg')

#The 1st element in this list gives us the image with the objects detected
results[0].show()