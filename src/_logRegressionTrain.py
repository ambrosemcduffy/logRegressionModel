import numpy as np
import cv2

from skimage.transform  import resize
import _logisticRegressionModel as logReg


def main(epochs, learningRate):
    
    xTrain, yTrain, xTest, yTest = logReg.getDataset(flattenImages=True, testSize=.1)
    print(yTrain)
    print(yTrain.shape)
    
    # Obtaining dimensions and Outputting
    print(xTrain.shape)
    nx, m = xTrain.shape
    ny = yTrain.shape[0]
    
    # parameters
    w, b = logReg.initializeParameters(nx)
    yhat = None
    yhatTest = None
    
    for epoch in range(epochs):
         yhat = logReg.sigmoid(logReg.forward(w, xTrain, b))
         error = logReg.crossEntropy(m, yTrain, yhat)
         if epoch % 100 == 0:
             
             yhatTest = logReg.sigmoid(logReg.forward(w, xTrain, b))
             errorTest = logReg.getAccuracy(yhatTest, yTrain)
             print("epoch: {} -- trainError: {} testAcc: {}".format(epoch, np.squeeze(error), np.squeeze(errorTest)))
         grads = logReg.backprop(yhat, yTrain, xTrain, m)
         w, b = logReg.optimization(learningRate, [w, b], grads)
         
    return (yhat, yTrain, yhatTest, yTest, w, b, m)


epochs=9100
yhat, yTrain, yhatTest, yTest, w, b, m = main(epochs=9100, learningRate=0.005)
print("Saving out weights..")
np.savez("weights/LogRegression_epoch{}.npz".format(epochs), **{"w":w, "b":b})
