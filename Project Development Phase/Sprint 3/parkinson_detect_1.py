#import the pakages

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import cv2
import os
import pickle

#Quatifying image
def quantify_image(image):
    #compute histogram of oriented gradients feature vector for the input image
    features=feature.hog(image,orientations=9,pixels_per_cell=(10,10),cells_per_block=(2,2),transform_sqrt=True,block_norm="L1")
    return features

def load_split(path):
    #grab list of images in the input dir,then initialize the list of data and class labels
    
    imagepaths=list(paths.list_images(path))
    data,labels=[],[]
    
    #loop over the image path
    for imagepath in imagepaths:
        #extract the class label from the filename
        label=imagepath.split(os.path.sep)[-2]
        
        #load the input image
        image=cv2.imread(imagepath)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image=cv2.resize(image,(200,200))
        image=cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        #quantify the image
        features=quantify_image(image)
        
        #update the data and labels
        data.append(features)
        labels.append(label)
        
    return (np.array(data),np.array(labels))
    
# define path to train and test dir

trainingpath= r"Desktop/dataset/spiral/training"
testingpath=r"Desktop/dataset/spiral/testing"

#loading train and test data

print("[INFO] loading data...")
(X_train,Y_train)=load_split(trainingpath)
(X_test,Y_test)=load_split(testingpath)
    
    
    
#Label Encoding
le=LabelEncoder()
Y_train=le.fit_transform(Y_train)
Y_test=le.transform(Y_test)
print(X_train.shape,Y_train.shape)

#Training The Model

print("[INFO] training model...")
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,Y_train)

#testing the model
testingpath=list(paths.list_images(testingpath))
idxs=np.arange(0,len(testingpath))
idxs=np.random.choice(idxs,size=(25,),replace=False)
images=[]

#loop over the testing samples
for i in idxs:
    image=cv2.imread(testingpath[i])
    output=image.copy()
        
    # load the input image,convert to grayscale and resize
    
    output=cv2.resize(output,(128,128))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.resize(image,(200,200))
    image=cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    #quantify the image and make predictions based on the  extracted feature using last trained random forest
    features=quantify_image(image)
    preds=model.predict([features])
    label=le.inverse_transform(preds)[0]
    #the set of output images
    if label=="healthy":
        color=(0,255,0)
    else:
        color=(0,0,255)
        
    cv2.putText(output,label,(3,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    images.append(output)

#creating a montage
montage=build_montages(images,(128,128),(5,5))[0]
cv2.imshow("Output",montage)
cv2.waitKey(0)

#model evaluation 
prediction=model.predict(X_test)
cm=confusion_matrix(Y_test,prediction).flatten()
print(cm)
(tn,fp,fn,tp)=cm
accuracy=(tp+tn)/float(cm.sum())
print(accuracy)

#storing the model

filename = 'parkinson.pkl'
pickle.dump(model, open(filename, 'wb'))
