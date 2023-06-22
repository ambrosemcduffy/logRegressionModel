import cv2
import numpy as np
from skimage.transform import resize
import _logisticRegressionModel as logReg
import argparse
import matplotlib.pyplot as plt

def predict_image(image_path):
    weights = np.load("weights/LogRegression_epoch9100.npz")
    w = weights["w"]
    b = weights["b"]

    _img = cv2.imread(image_path)
    img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    img = logReg.resizeImage(img, [64, 64])

    img = img.reshape(img.shape[0] * img.shape[1] * img.shape[2], 1)
    z = logReg.forward(w, img, b)
    pred = logReg.sigmoid(z)
    print(str(pred[0][0] * 100)[:4]+"%")

    # Apply a threshold to get a binary prediction
    threshold = 0.6
    binary_prediction = (pred > threshold).astype(int)
    if binary_prediction[0][0] == 1:
        title = "Binary prediction: Hm.. I think this is a Cat!" 
    else:
        title = "Binary prediction: Wait.. this is not a Cat.\n Something else."

    plt.imshow(_img)
    plt.title(title)
    plt.show()
    return binary_prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image prediction using logistic regression")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    prediction = predict_image(args.image_path)
    
