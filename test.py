from ultralytics import YOLO
if __name__ == '__main__':
# Load a model
    model = YOLO('path of the trained model')  # load a custom model
    # Predict with the model
    results = model('path of the infer data', save = True)  # predict on an image

    