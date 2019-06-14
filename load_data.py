import os
import cv2 
import numpy as np 
import pickle 
from Data_Augmentation import agumentation
import pandas as pd
    

def load():

    Img_Data = []
    Label = []
    id = []
    pathFile = []
    count = 0
    rootFolder = 'dataset'
    label_csv = []

    for folder_root in os.listdir(rootFolder):
        
        folderName = os.path.join(rootFolder, folder_root)

        print("Loading folder .... ", folderName)
        for folder in os.listdir(folderName):

            folderFile = os.path.join(folderName, folder)
            print("Loading folder...", folder)
            
            for fileName in os.listdir(folderFile):

                fileImage = fileName
                fileName = os.path.join(folderFile, fileName)
                print(fileName)

                img = cv2.imread(fileName)
                if img is not None:
		    
                    img = cv2.resize(img, (224,224))
                    Img_Data.append(img)
                    agu_img = agumentation(img)

                    Img_Data = Img_Data +  agu_img

                    Label.append(folder)
                    labels = [folder]* len(agu_img)
                    Label = Label + labels

                    id.append(count)
                    count += 1
                    pathFile.append(fileImage)
                    label_csv.append(folder)


        print("Done")
    print('Number of data: ', len(Img_Data))
    print('Number of labels: ', len(Label))
    dataFrame = {'ID':id, "Name" : pathFile, 'Label' :label_csv}
    df = pd.DataFrame(dataFrame)
    df.to_csv('data.csv')
    return (Img_Data, Label)


load()


