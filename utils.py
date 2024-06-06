import os,shutil,glob
import pandas as pd
import random
import matplotlib.pyplot as plt
from functools import reduce
import cv2
import pickle
from imblearn.over_sampling import SMOTE
import numpy as np
from scipy.ndimage import zoom
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.model_selection import StratifiedKFold,cross_validate,GridSearchCV
import seaborn as sns
import xgboost as xgb
def loadCommonData() :
    input_folder = '/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/Hima'
    features = ['B04B','B05B','B06B','B09B','B10B','B11B',
                'B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB']
    dict_map_namefile_to_data = {
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

    months = ['04','10']
    for month in months :
        print('Months :',month)
        for band in features:
            path_month = input_folder + '/'+band+'/2019/'+month
            listname_day = os.listdir(path_month)
            listname_day.sort()
            list_name_file = []
            for day in listname_day :
                path_day = path_month + '/' + day
                listfile_day = os.listdir(path_day)
                listfile_day.sort()
                for file in listfile_day :
                    list_name_detail = file.split('_')
                    list_name_file.append(list_name_detail[1])
            name_temp_data = dict_map_namefile_to_data[band]
            for k in list_name_file :
                name_temp_data.append(k)
        # dict_map_feature_to_data[band] = band_data
    print('**************')
    sets = [set(dict_map_namefile_to_data[k]) for k in features]
    common_elemetns = reduce(lambda s1,s2 : s1 & s2,sets)
    print('Common :',len(common_elemetns))
    print("****************")
    return common_elemetns,dict_map_namefile_to_data

def getSampleData(namefile) :
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

    months = ['04','10']

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
                    list_name_file = file.split('_')
                    name_ = list_name_file[1]
                    name_ = name_.replace('.Z','')
                    if name_ == namefile :
                        name_file = path_day + '/' + file
                        image = cv2.imread(name_file, cv2.IMREAD_UNCHANGED)
                        band_data.append(image)
                        break
            band_temp_data = dict_map_feature_to_data[band]
            for k in band_data :
                band_temp_data.append(k)
    for i in dict_map_feature_to_data :
        print(i, '    :',len(dict_map_feature_to_data[i]))
    return dict_map_feature_to_data

def prepareInputData() :
    input_folder = '/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/Hima'
    features = ['B04B','B05B','B06B','B09B','B10B','B11B',
                'B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB']

    common_elemetns,dict_map_namefile_to_data= loadCommonData()

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

    months = ['04','10']

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
                    list_name_file = file.split('_')
                    name_ = list_name_file[1]
                    if name_ in common_elemetns :
                        name_file = path_day + '/' + file
                        image = cv2.imread(name_file, cv2.IMREAD_UNCHANGED)
                        band_data.append(image)
            band_temp_data = dict_map_feature_to_data[band]
            for k in band_data :
                band_temp_data.append(k)

    for i in dict_map_feature_to_data :
        print(i, '    :',len(dict_map_feature_to_data[i]))
    return dict_map_feature_to_data


def prepareOutputData() :
    elements_common,dict_ = loadCommonData()
    element_com = []
    for i in elements_common :
        i = i.replace('.Z','') 
        element_com.append(i)
    months = ['04','10']
    list_image_output = []
    for month in months :
        output_folder = '/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/Precipitation/AWS/2019/'+month
        listnameoutputfolder = os.listdir(output_folder)
        listnameoutputfolder.sort()
        for day in listnameoutputfolder :
            path_day = output_folder + '/' + day
            listnamefile = os.listdir(path_day)
            listnamefile.sort()
            for name in listnamefile :
                list_name_file = name.split('_')
                name_ = list_name_file[1]
                
                name_ = name_[:-6]
                print('name_ :',name_)
                if name_ in element_com :
                    print("name_ :",name_)
                    image_name = path_day + '/' + name
                    image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
                    list_image_output.append(image)
    min_length  = 408
    list_image_output = list_image_output * (min_length // len(list_image_output)) + random.sample(list_image_output, min_length % len(list_image_output))
    print(len(list_image_output))
    return list_image_output
def mergeDataInput() :
    comon,dict_ = loadCommonData()
    number_data = len(comon)
    print('number data :',number_data)
    dict_map_feature_to_data = prepareInputData()
    features = ['B04B','B05B','B06B','B09B','B10B','B11B',
                'B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB']
    list_data_image = []
    for i in range(0,number_data) :
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

    for i in range(1,number_data) :
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
    comon,dict_ = loadCommonData()
    number_data = len(comon)
    list_image_output = prepareOutputData()
    ima_out = list_image_output[0]

    ima_out = np.array(ima_out)
    rows,cols = ima_out.shape
    ima_out = ima_out.reshape((1,rows*cols))
    ima_out = ima_out.transpose()
    ima_out.shape
    df_output = pd.DataFrame(ima_out,columns = ['Output'])

    for i in range(1,number_data) :
        tmp_output = list_image_output[i]
        tmp_output = np.array(tmp_output)
        rows,cols = tmp_output.shape
        tmp_output = tmp_output.reshape((1,rows*cols))
        tmp_output = tmp_output.transpose()
        df_output1 = pd.DataFrame(tmp_output,columns = ['Output'])
        df_output = pd.concat([df_output,df_output1],axis=0)
    
    df_output['Output'] = df_output['Output'].apply(lambda x: 0 if x<0 else 1)
    print(df_output)
    # writeToCSV(df_output,'/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/dataOutput.csv')
def writeToCSV(dataframe,file) :
    dataframe.to_csv(file,index = False)

def printDetailsReport(y_true,y_pred) :
    print(classification_report(y_true,y_pred))

def loadTrainTestData() :
    data_features = pd.read_csv('/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/dataInput.csv')
    data_output = pd.read_csv('/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/dataOutput.csv')
    # print(data_output['Output'].value_counts())
    sm  =SMOTE()
    X_sm,Y_sm = sm.fit_resample(data_features,data_output)
    # print(X_sm.shape,Y_sm.shape) 
    print(Y_sm.value_counts())
    X_train,X_test,y_train,y_test = train_test_split(X_sm,Y_sm,test_size=0.2)
    print("Xtrain, Xtest :",X_train.shape,', ',X_test.shape)
    return X_train,X_test,y_train,y_test
def loadModelDecisionTree() :
    X_train,X_test,y_train,y_test = loadTrainTestData()

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    print("Ready to train!!!!!")
    clf = DecisionTreeClassifier(random_state=0,max_depth=10,class_weight= 'balanced')
    clf.fit(X_train,y_train)
    pred_test = clf.predict(X_test)

    print("Accuracy score :",accuracy_score(y_test,pred_test))
    print("Precision score :",precision_score(y_test,pred_test))
    print("Recall Score :",recall_score(y_test,pred_test))
    print("Confu matrics :")
    print(confusion_matrix(y_test,pred_test))
    print('Report :')
    printDetailsReport(y_test,pred_test)
    print("Model.score : ",clf.score(X_test,y_test))
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
    print('readyyy')
    param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1]
}

    skf = StratifiedKFold(n_splits=2,shuffle=True)
    model = xgb.XGBClassifier(random_state = 42)
    grid = GridSearchCV(model,param_grid=param_grid,cv = skf,scoring='f1',verbose=3,n_jobs=-1)
    grid.fit(X_train,y_train)
    print("Best params :",grid.best_params_)
    print("Best score :",grid.best_score_)
    model = grid.best_estimator_

    print('**********')
    pred_test = model.predict(X_test)
    
    print("Accuracy score :",accuracy_score(y_test,pred_test))
    print("Precision score :",precision_score(y_test,pred_test))
    print("Recall Score :",recall_score(y_test,pred_test))
    print("Confu matrics :")
    print(confusion_matrix(y_test,pred_test))
    print('Report :')
    printDetailsReport(y_test,pred_test)
    print("Model.score : ",model.score(X_test,y_test))



    saveTheBestModel(model=model)
    print("save successfully")
def saveBarChartImage(data,link) :
    fig,ax = plt.subplots()
    a = sns.countplot(x = data)
    fig.savefig(link)
    plt.close()
def saveDensity(name,data,link) :
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True)
    plt.title(f'Density Plot for {name}')
    plt.xlabel('B04B')
    plt.ylabel('Density')

    # Lưu biểu đồ
    plt.savefig(link)
    plt.close()
def savePieChertImage(data) :
# Đếm số lượng mỗi giá trị (0 và 1)
    counts = data.value_counts()

    # Tạo nhãn và giá trị cho biểu đồ tròn
    labels = ['0', '1']
    sizes = [counts.get(0, 0), counts.get(1, 0)]
    print(sizes)
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # "explode" là phần nổi của mỗi miếng bánh

    # Vẽ biểu đồ tròn
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)

    fig.savefig('/home/quoc/works/Learn/learnLLMs/AI_classification/pieOutput.png', bbox_inches='tight', pad_inches=0)

def loadXGBModel() :
    X_train,X_test,y_train,y_test = loadTrainTestData()

    
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    model_xgb = xgb.XGBClassifier(random_state = 42,n_estimators = 100)
    model_xgb.fit(X_train,y_train)
    pred_test = model_xgb.predict(X_test)
    
    print("Accuracy score :",accuracy_score(y_test,pred_test))
    print("Precision score :",precision_score(y_test,pred_test))
    print("Recall Score :",recall_score(y_test,pred_test))
    print("Confu matrics :")
    print(confusion_matrix(y_test,pred_test))
    print('Report :')
    printDetailsReport(y_test,pred_test)
    print("Model.score : ",model_xgb.score(X_test,y_test))

def loadRandomForestModel() :
    X_train,X_test,y_train,y_test = loadTrainTestData()

    
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    model_rf = xgb.XGBClassifier(random_state = 42,n_estimators = 100)
    model_rf.fit(X_train,y_train)
    pred_test = model_rf.predict(X_test)
    print("Accuracy score :",accuracy_score(y_test,pred_test))
    print("Precision score :",precision_score(y_test,pred_test))
    print("Recall Score :",recall_score(y_test,pred_test))
    print("Confu matrics :")
    print(confusion_matrix(y_test,pred_test))
    print('Report :')
    printDetailsReport(y_test,pred_test)
    print("Model.score : ",model_rf.score(X_test,y_test))

# def predict(model,infor_image_band) :
    
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
# loadTrainTestData()


