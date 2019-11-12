from LogisticRegression import *

'''
测试函数
'''
def test_logistic_regression():
     path="D:\\"
     training_sample = 'trainingSet.txt'  #训练数据文件
     testing_sample = 'testingSet.txt'    #测试训练文件
     trainingSet, trainingLabel = loadDataSet(path,training_sample) #读入训练数据
     A = gradAscent(trainingSet, trainingLabel)   # 回归系数a的值
     testingSet, testingLabel = loadDataSet(path, testing_sample) #读入测试数据
     h = getResult(testingSet,A) #预测结果
     plotBestFit(A.getA(),testingSet,testingLabel,h)  #图形化展示

'''
程序入口
'''
if __name__ == "__main__":
     test_logistic_regression()