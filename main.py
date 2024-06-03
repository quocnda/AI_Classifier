import os,shutil,glob
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.model_selection import StratifiedKFold,cross_validate,GridSearchCV

def prepareInputData() :
    input_folder = '/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/Hima'
    features = ['B04B','B05B','B06B','B09B','B10B','B11B',
                'B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB']
    dict_map_feature_to_data = {
        "B04B" : [],
        "B05B" : [],
        'B06B' : [],
        'B09B' : [],
        'B10B' : [],
        'B11B' : [],
                'B12B' : [],
        'B14B' : [],
        'B16B' : [],
        'I2B' : [],
        'I4B' : [],
        'IRB' : [],
        'VSB' : [],
        'WVB' : []
    }

    months = ['04']


    for month in months :
        print('Months :',month)
        for band in features:
            path_month = input_folder + '/'+band+'/2019/'+month
            listname_day = os.listdir(path_month)
            listname_day.sort()
            band_data = []
            for day in listname_day :
                path_day = path_month + '/' + day
                listfile_day = os.listdir(path_day)
                listfile_day.sort()
                for file in listfile_day :
                    name_file = path_day + '/' + file
                    image = cv2.imread(name_file, cv2.IMREAD_UNCHANGED)
                    band_data.append(image)
            band_temp_data = dict_map_feature_to_data[band]
            for k in band_data :
                band_temp_data.append(k)
        # dict_map_feature_to_data[band] = band_data

    for i in features :
        k = dict_map_feature_to_data[i]
        print(i, "   :",len(k))
    print("****************")


    max_length = max([len(data) for data in dict_map_feature_to_data.values()])

    for band in features:
        band_data = dict_map_feature_to_data[band]
        if len(band_data) < max_length:
            scale_factor = max_length / len(band_data)
            band_data = zoom(band_data, zoom=[scale_factor, 1, 1], order=0)
            dict_map_feature_to_data[band] = band_data[:max_length]

    for i in features :
        k = dict_map_feature_to_data[i]
        print(i, "   :",len(k))
    return dict_map_feature_to_data
def prepareOutputData() :
    import random
    months = ['04']
    list_image_output = []
    for month in months :
        output_folder = '/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/Precipitation/Radar/2019/'+month
        listnameoutputfolder = os.listdir(output_folder)
        listnameoutputfolder.sort()
        for day in listnameoutputfolder :
            path_day = output_folder + '/' + day
            listnamefile = os.listdir(path_day)
            listnamefile.sort()
            for name in listnamefile :
                image_name = path_day + '/' + name
                image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
                list_image_output.append(image)
    min_length  = 594
    list_image_output = list_image_output * (min_length // len(list_image_output)) + random.sample(list_image_output, min_length % len(list_image_output))
    print(len(list_image_output))
    return list_image_output
def mergeDataInput() :
    dict_map_feature_to_data = prepareInputData()
    features = ['B04B','B05B','B06B','B09B','B10B','B11B',
                'B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB']
    list_data_image = []
    for i in range(0,593) :
        data_image = []
        for feature in features :
            image = dict_map_feature_to_data[feature][i]
            data_image.append(image)
        list_data_image.append(data_image)

    tmp = list_data_image[0]
    tmp = np.array(tmp)
    n_bands,rows,cols = tmp.shape
    tmp = tmp.reshape((n_bands,rows*cols))
    tmp = tmp.transpose()
    df = pd.DataFrame(tmp,columns = features)

    for i in range(1,593) :
        print(i)
        tmp1 = list_data_image[i]
        tmp1 = np.array(tmp1)
        n_bands,rows,cols = tmp1.shape
        tmp1 = tmp1.reshape((n_bands,rows*cols))
        tmp1 = tmp1.transpose()
        df1 = pd.DataFrame(tmp1,columns = features)
        df = pd.concat([df,df1],axis=0)
    print(df)
    # writeToCSV(df,'/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/dataInput.csv')

def mergeDataOutput() :
    list_image_output = prepareOutputData()
    ima_out = list_image_output[0]

    ima_out = np.array(ima_out)
    rows,cols = ima_out.shape
    ima_out = ima_out.reshape((1,rows*cols))
    ima_out = ima_out.transpose()
    ima_out.shape
    df_output = pd.DataFrame(ima_out,columns = ['Output'])

    for i in range(1,593) :
        print(i)
        tmp_output = list_image_output[i]
        tmp_output = np.array(tmp_output)
        rows,cols = tmp_output.shape
        tmp_output = tmp_output.reshape((1,rows*cols))
        tmp_output = tmp_output.transpose()
        df_output1 = pd.DataFrame(tmp_output,columns = ['Output'])
        df_output = pd.concat([df_output,df_output1],axis=0)
    # writeToCSV(df_output,'/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/dataOutput.csv')
    print(df_output)
def writeToCSV(dataframe,file) :
    dataframe.to_csv(file,index = False)

def loadTrainTestData() :
    data_features = pd.read_csv('/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/dataInput.csv')
    data_output = pd.read_csv('/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/dataOutput.csv')
    data_output['Output'] = data_output['Output'].apply(lambda x: 0 if x<0 else 1)

    X_train,X_test,y_train,y_test = train_test_split(data_features,data_output,test_size=0.2)
    return X_train,X_test,y_train,y_test
def loadModelAndPrepare() :
    X_train,X_test,y_train,y_test = loadTrainTestData()
    print("Ready to train!!!!!")
    clf = DecisionTreeClassifier(random_state=0,max_depth=10)
    clf.fit(X_train,y_train)
    pred_train = clf.predict(X_train)
    print("Predict train data : ",pred_train )

    print("Accuracy score :",accuracy_score(y_train,pred_train))
    print("Precision score :",precision_score(y_train,pred_train))
    print("Recall Score :",recall_score(y_train,pred_train))
    print("Confu matrics :")
    print(confusion_matrix(y_train,pred_train))

    saveTheModel(clf)
    print("Save the model successfully!!!")
def saveTheModel(model) :
    file_name = '/home/quoc/works/Learn/learnLLMs/AI_classification/myModel.sav'
    pickle.dump(model,open(file_name,'wb'))
def saveTheBestModel(model) :
    file_name = '/home/quoc/works/Learn/learnLLMs/AI_classification/myBestModel.sav'
    pickle.dump(model,open(file_name,'wb'))
def loadTheModel() :
    file = '/home/quoc/works/Learn/learnLLMs/AI_classification/myModel.sav'
    model = pickle.load(open(file,'rb'))
    return model
def K_FoldCross(X,y) :
    skf = StratifiedKFold(n_splits=3,shuffle=True)
    model = DecisionTreeClassifier(max_depth=10)
    results = cross_validate(
        model,X,y,cv = skf,
        scoring=('accuracy','f1')
    )
    print("Result :",results)
def gridSearchCV() :
    X_train,X_test,y_train,y_test = loadTrainTestData()
    param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [8,14]
    }
    skf = StratifiedKFold(n_splits=3,shuffle=True)
    model = DecisionTreeClassifier()
    grid = GridSearchCV(model,param_grid=param_grid,cv = skf,scoring='f1',verbose=3,n_jobs=-1)
    grid.fit(X_train,y_train)
    print("Best params :",grid.best_params_)
    print("Best score :",grid.best_score_)
    model = grid.best_estimator_
    saveTheBestModel(model=model)
    print("save successfully")
def main() :
    gridSearchCV()
# *********************Predict for the test data**************
    # X_train,X_test,y_train,y_test = loadTrainTestData()
    # model = loadTheModel()
    # pred_test = model.predict(X_test)
    # print("Accuracy score :",accuracy_score(y_test,pred_test))
    # print("Precision score :",precision_score(y_test,pred_test))
    # print("Recall Score :",recall_score(y_test,pred_test))
    # print("Confu matrics :")
    # print(confusion_matrix(y_test,pred_test))
main()