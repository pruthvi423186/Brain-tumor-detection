from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  

# Run batched inference on a list of images
results = model(["image1.jpg"], stream=True)  

# Process results generator
for result in results:
    boxes = result.boxes  
    masks = result.masks  
    keypoints = result.keypoints  
    probs = result.probs 
    obb = result.obb  
    result.show()  
    result.save(filename="result.jpg") 
