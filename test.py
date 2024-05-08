from ultralytics import YOLO
if __name__ == '__main__':
# Load a model
    model = YOLO('C:/Users/User/Desktop/YOLOV8/runs/obb/train44/weights/best.pt')  # load a custom model
    # Predict with the model
    results = model('C:/Users/User/Desktop/Internship_Crimson/data/', save = True)  # predict on an image

    