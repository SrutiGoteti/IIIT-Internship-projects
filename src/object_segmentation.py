from ultralytics import YOLO

# Load pretrained model for segmentation
model = YOLO('yolov8n-seg.pt')

# List of image URLs
images = [
    'https://thumbs.dreamstime.com/b/cybertruck-drives-busy-washington-d-c-intersection-traffic-signals-usa-may-futuristic-stainless-steel-truck-384528985.jpg',
    'https://ultralytics.com/images/zidane.jpg',
    'https://ultralytics.com/images/bus.jpg'
]

# Run segmentation on each image
for img in images:
    results = model(img)
    print(f"Segmentation Results for {img}:")
    results[0].show()
