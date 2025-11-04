from ultralytics import YOLO

# Load pretrained model for detection
model = YOLO('yolov8n.pt')

# List of image URLs
images = [
    'https://ultralytics.com/images/zidane.jpg',
    'https://ultralytics.com/images/bus.jpg',
    'https://thumbs.dreamstime.com/b/cybertruck-drives-busy-washington-d-c-intersection-traffic-signals-usa-may-futuristic-stainless-steel-truck-384528985.jpg'
]

# Run detection on each image
for img in images:
    results = model(img)
    print(f"Results for {img}:")
    results[0].show()
