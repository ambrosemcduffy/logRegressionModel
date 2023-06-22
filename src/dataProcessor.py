import os

from skimage.transform  import resize
import numpy as np
import cv2
import matplotlib.pyplot as plt

import h5py

def getVectorizeImage(path):
    imageArray, labelData = importImages(path)
    x_flatten = imageArray.reshape(imageArray.shape[0], -1).T
    XData = x_flatten /255.
    YData = labelData.reshape(labelData.shape[0], 1).T
    return (XData, YData)

def importImages(path, crop=[64, 64]):
    print("importing images")
    imageArray = []
    label = []
    if os.path.exists(path) is not True:
        return None
    
    for folderName in sorted(os.listdir(path)):
        for imageName in sorted(os.listdir(path+"/"+folderName+"/")):
            fullPath =path+"/"+folderName+"/" + imageName
            try:
                image = cv2.imread(fullPath)
                if image is not None:
                    newImage = resizeImage(image, crop)
                    if imageName.endswith(".jpg") or imageName.endswith(".jfif"):
                        if folderName == "cat" or folderName == "cats":
                            label.append(1)
                        elif folderName == "dog" or folderName == "puppies":
                            label.append(2)
                        elif folderName == "chicks":
                            label.append(3)
                        elif folderName == "people" or folderName == "peoplePortraits":
                            label.append(4)
                        else:
                            label.append(0)
                        imageArray.append(newImage)
            except cv2.error as e:
                print(f"Error reading image {fullPath}: {e}")
    return (np.array(imageArray), np.array(label))


def resizeImage(image, dims):
        new_height, new_width = getNewImageSize(image, dims)
        newImage = resize(image, (new_width, new_height))
        return newImage[: dims[0], : dims[1]]


def getNewImageSize(image, dims):
        width, height, color = image.shape
        if height > width:
            new_height, new_width = (int(dims[0] * height / width), dims[0])
        elif height < width:
            new_height, new_width = (dims[0], int(dims[0] * width / height))
        else:
            new_height, new_width = (dims[0], dims[0])
        return new_height, new_width


def processImage(path):
    image = cv2.imread(path)
    newImage = resizeImage(image, [64, 64])
    print(newImage.shape)
    image_flat = newImage.reshape(newImage.shape[0] * newImage.shape[1] * 3, 1)
    X = image_flat /255.
    return X


def displayImages(images, labels, size=(2, 2)):
    fig, ax = plt.subplots(size[0], size[1], figsize=(10,10))
    for i in range(size[1]):
        for j in range(size[0]):
            image = images[i*size[1]+j]
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax[i, j].imshow(image)
            ax[i, j].axis('off')
            ax[i, j].set_title(labels[i*size[1]+j], fontsize=10, y=-0.25 )
    fig.tight_layout()
    plt.show()


def getDataset(flattenImages=False, testSize=.1):
    with h5py.File('data.hdf5', 'r') as f:
    # Access the 'images' and 'labels' datasets and read their contents
        xtrain = f['images'][:]
        ytrain = f['labels'][:]
        
        # shuffle
        idx = np.random.permutation(xtrain.shape[0])
        xtrain, ytrain = xtrain[idx], ytrain[idx]
        
        # Flatten the input data
        testSize = int(xtrain.shape[0] * testSize)

        xtrain = xtrain[testSize:]
        ytrain = ytrain[testSize:]
        ytrain = ytrain.reshape(ytrain.shape[0], 1)

        xtest = xtrain[:testSize]
        ytest = ytrain[:testSize]
        ytest = ytest.reshape(ytest.shape[0], 1)

        if flattenImages:
            x_flatten = xtrain.reshape(xtrain.shape[0], -1).T
            y_flatten = ytrain.reshape(ytrain.shape[0], -1).T


            x_Testflatten = xtest.reshape(xtest.shape[0], -1).T
            y_Testflatten = ytest.reshape(ytest.shape[0], -1).T
            return x_flatten, y_flatten, x_Testflatten, y_Testflatten
    
    return xtrain, ytrain, xtest, ytest


def saveData(images, labels):
    with h5py.File('data.hdf5', 'w') as f:
        images_dset = f.create_dataset('images', shape=images.shape)
        labels_dset = f.create_dataset('labels', shape=labels.shape, dtype=np.uint8)
        images_dset[:] = images
        labels_dset[:] = labels


def displayPrediction(data, yhat, dims=(4, 4)):
    threshold = 0.05
    yhat_binary = np.where(yhat >= threshold, 1, 0)
    displayImages(data, yhat[0], size=dims)


def unFlattenImages(images):
    imageArr = []
    for i in range(images.shape[1]):
        imageArr.append(images[:,i].reshape(64, 64, 3))
    return np.array(imageArr)

