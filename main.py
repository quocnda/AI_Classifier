from utils import *
import seaborn as sns
from imblearn.over_sampling import SMOTE
def main() :
    # print("Decision Tree Model !!!")
    # loadModelDecisionTree()
    # print('******')
    # print("xGB Model !!!")
    # loadXGBModel()
    # print("*******")
    # print('Random Forest Model')
    # loadRandomForestModel()
    
    data_frame_pixel = getSampleData('201904230200')
    model = loadTheModel('/home/quoc/works/Learn/learnLLMs/AI_classification/myBestModel.sav')
    a = model.predict(data_frame_pixel)
    a = a.reshape(90,250)
    print(a)
    plt.imshow(a,cmap = 'gray')
    plt.savefig('/home/quoc/works/Learn/learnLLMs/AI_classification/ImageChart/t.png')
    plt.close()

    # data = getFinalData()
    # a = data['Output'].value_counts()
    # print(a)
    # savePieChertImage(data['Output'])
    # prepareOutputData()

    # mergeDataOutput()
    

    # data = getFinalData()
    # data_features = data.drop(columns=['Output'])
    # data_out = data['Output']
    # # print(dataOutput.value_counts())
    # sm  =SMOTE()
    # X_sm,Y_sm = sm.fit_resample(data_features,data_out)
    # K_FoldCross(X_sm,Y_sm)
    


    # gridSearchCV()



    # loadTrainTestData()



    # features = ['B04B','B05B','B06B','B09B','B10B','B11B',
    #             'B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB']
    # data_features = pd.read_csv('/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/dataInput.csv')   
    # for i in features :
    #     saveDensity(i,data_features[i],f'/home/quoc/works/Learn/learnLLMs/AI_classification/ImageChart/{i}.png')



    # data_features = pd.read_csv('/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/dataInput.csv')
    # model = loadTheModel()
    # datta = data_features.values
    # print(datta[0:10])
    # # model.predict(datta[0].reshape(1,-1))
    # predict(model,datta[0:10])
    


    # data_output = pd.read_csv('/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/dataOutput.csv')
    # data_features = pd.read_csv('/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/dataInput.csv')
    # print(data_output['Output'].value_counts())
    # sm  =SMOTE()
    # X_sm,Y_sm = sm.fit_resample(data_features,data_output)
    # print(X_sm.shape,Y_sm.shape) 
    # savePieChertImage(Y_sm)
main()

# dataOut = pd.read_csv('/home/quoc/works/Learn/learnLLMs/data/DATAForBTL/DATA_SV/dataOutput.csv')
# # Đếm số lượng mỗi giá trị (0 và 1)
# counts = dataOut['Output'].value_counts()

# # Tạo nhãn và giá trị cho biểu đồ tròn
# labels = ['0', '1']
# sizes = [counts.get(0, 0), counts.get(1, 0)]
# colors = ['#66b3ff', '#ff9999']
# explode = (0.1, 0)  # "explode" là phần nổi của mỗi miếng bánh

# # Vẽ biểu đồ tròn
# fig, ax = plt.subplots()
# ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)

# fig.savefig('/home/quoc/works/Learn/learnLLMs/AI_classification/pie.png', bbox_inches='tight', pad_inches=0)



    # print(a)
    # fig.savefig('/home/quoc/works/Learn/learnLLMs/AI_classification/anh.png', bbox_inches='tight', pad_inches=0)


    # plt.close(fig)