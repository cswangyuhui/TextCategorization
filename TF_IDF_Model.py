# -*- coding: utf-8 -*-
import os
import datetime
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

reload(sys)
sys.setdefaultencoding('utf-8')


class TF_IDF:
    data_path = "/Users/wangyuhui/Desktop/dataminingData/train/"
    test_path = "/Users/wangyuhui/Desktop/dataminingData/test/"
    result_path = "/Users/wangyuhui/Desktop/dataminingResult1/"
    stop_words_path = "/Users/wangyuhui/Desktop/dataminingData/stop_words_ch.txt"
    #class_path = "../test_res/"
    stop_words = []
    class_code = {"healthy":0,"history":1,"education":2,"ent":3,
                  "food":4,"houseproperty":5,"military":6,"sport":7,
                  "stock":8,"technology":9}

    def __init__(self):
       self.ConfusionMatrix = np.zeros([len(TF_IDF.class_code),len(TF_IDF.class_code)])
       TF_IDF.stop_words = self.read_file(TF_IDF.stop_words_path).strip().split("\n")

    def printList(self, mylist):
        for item in mylist:
            print item,
        print

    # 写文件
    def save_file(self, path, content):
        with open(path, 'w') as f:
            f.write(content)

    # 读文件
    def read_file(self, path):
        with open(path, 'r') as f:
            return f.read()
    #读取bunch文件
    def readbunchobj(self,path):
        with open(path, "rb") as file_obj:
            bunch = pickle.load(file_obj)
        return bunch

    def writebunchobj(self,path, bunchobj):
        with open(path, "wb") as file_obj:
            pickle.dump(bunchobj, file_obj)

    # 加载分词结果文件
    def loadSegmentation(self,path):
        begintime = datetime.datetime.now()
        fileDocs = os.listdir(path)
        print fileDocs
        for item in fileDocs:
            if item.startswith("."):
                fileDocs.remove(item)
        print fileDocs
        bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
        bunch.target_name.extend(fileDocs)
        # 获取每个目录下所有的文件
        for mydir in fileDocs:
            class_path = path + mydir + "/"  # 拼出分类子目录的路径
            file_list = os.listdir(class_path)  # 获取class_path下的所有文件
            for file_path in file_list:  # 遍历类别目录下文件
                if file_path.endswith("txt") == False:
                    continue
                fullname = class_path + file_path  # 拼出文件名全路径
                bunch.label.append(mydir)
                bunch.contents.append(self.read_file(fullname))  # 读取文件内容
                bunch.filenames.append(mydir+"/"+file_path)
        self.writebunchobj(TF_IDF.result_path+"trainData.dat",bunch)
        endtime = datetime.datetime.now()
        span = endtime - begintime
        print "训练bunch:contents长度、label长度:",len(bunch.contents),len(bunch.label)
        print "训练数据保存完成,所花费时间为",span.seconds
    def loadTestData(self, path):
        begintime = datetime.datetime.now()
        fileDocs = os.listdir(path)
        print fileDocs
        for item in fileDocs:
            if item.startswith("."):
                fileDocs.remove(item)
        print fileDocs
        bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
        bunch.target_name.extend(fileDocs)
        # 获取每个目录下所有的文件
        for mydir in fileDocs:
            class_path = path + mydir + "/"  # 拼出分类子目录的路径
            file_list = os.listdir(class_path)  # 获取class_path下的所有文件
            for file_path in file_list:  # 遍历类别目录下文件
                if file_path.endswith("txt") == False:
                    continue
                fullname = class_path + file_path  # 拼出文件名全路径
                bunch.label.append(mydir)
                bunch.contents.append(self.read_file(fullname))  # 读取文件内容
                bunch.filenames.append(mydir + "/" + file_path)
        self.writebunchobj(TF_IDF.result_path + "testData.dat", bunch)
        endtime = datetime.datetime.now()
        span = endtime - begintime
        print "测试bunch:contents长度、label长度:", len(bunch.contents), len(bunch.label)
        print "测试数据保存完成,所花费时间为",span.seconds

    def calculateTFIDF(self,train_tfidf_path,bunch_path,tfidf_path):
        begintime = datetime.datetime.now()
        bunch = self.readbunchobj(bunch_path)
        tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tfidfmetrix=[],
                           vocabulary={})

        if train_tfidf_path is not None:
            trainbunch = self.readbunchobj(train_tfidf_path)
            tfidfspace.vocabulary = trainbunch.vocabulary
            vectorizer = TfidfVectorizer(stop_words=TF_IDF.stop_words, sublinear_tf=True, max_df=0.5,
                                         vocabulary=trainbunch.vocabulary)
            tfidfspace.tfidfmetrix = vectorizer.fit_transform(bunch.contents)

        else:
            vectorizer = TfidfVectorizer(stop_words=TF_IDF.stop_words, sublinear_tf=True, max_df=0.5)
            tfidfspace.tfidfmetrix  = vectorizer.fit_transform(bunch.contents)
            tfidfspace.vocabulary = vectorizer.vocabulary_
            self.writebunchobj(TF_IDF.result_path + "myvocabulary.dat", vectorizer.vocabulary_)
            # ch2 = SelectKBest(chi2, k=130000)
            # train_X = ch2.fit_transform()

        self.writebunchobj(tfidf_path, tfidfspace)
        endtime = datetime.datetime.now()
        span = endtime - begintime
        if train_tfidf_path is not None:
            print "生成测试tf-idf矩阵,所花费时间为",span.seconds
        else:
            print "生成训练tf-idf矩阵,所花费时间为",span.seconds

    def metrics_result(self,actual, predict):
        print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
        print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
        print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))
        print(classification_report(testSet.label, predict))

    def trainProcess(self,trainSet):
        print "朴素贝叶斯训练过程开始:"
        begintime = datetime.datetime.now()
        # 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高
        clf = MultinomialNB(alpha=0.001).fit(trainSet.tfidfmetrix, trainSet.label)
        with open(TF_IDF.result_path+"MultinomialNBModel.dat", "wb") as file_obj:
            pickle.dump(clf, file_obj)
        endtime = datetime.datetime.now()
        span = endtime - begintime
        print "朴素贝叶斯训练时间:",span.seconds

    def SVMProcess(self, trainSet):
        print "SVM训练过程开始:"
        begintime = datetime.datetime.now()
        clf = svm.SVC(C=1.0, cache_size=800, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
                      gamma='auto', kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
        clf.fit(trainSet.tfidfmetrix, trainSet.label)
        with open("/Users/wangyuhui/Desktop/svmResult/svmModel.dat", "wb") as file_obj:
        # with open(TF_IDF.result_path + "svmModel.dat", "wb") as file_obj:
            pickle.dump(clf, file_obj)
        endtime = datetime.datetime.now()
        span = endtime - begintime
        print "SVM训练时间:", span.seconds
        test_tfidf_path = TF_IDF.result_path + "test_tfidf_svm.dat"
        testSet = TF_IDF.readbunchobj(test_tfidf_path)
        self.predictProcess("/Users/wangyuhui/Desktop/svmResult/svmModel.dat",testSet)

    def predictProcess(self,modelPath,testSet):
        if modelPath is None:
            print "模型路径未知"
            return
        clf = self.readbunchobj(modelPath)
        errorFile = "MultinomialNBerror.txt"
        if modelPath.find("svm") != -1:
            errorFile = "SVMerror.txt"
        print "加载模型成功"
        # 预测分类结果
        predicted = clf.predict(testSet.tfidfmetrix)
        outPutList = []
        for flabel, file_name, expct_cate in zip(testSet.label, testSet.filenames, predicted):
            self.ConfusionMatrix[TF_IDF.class_code[flabel]][TF_IDF.class_code[expct_cate]] += 1
            if flabel != expct_cate:
                outPutList.append(file_name + ": 实际类别:" + flabel + " -->预测类别:" + expct_cate)
        self.save_file(TF_IDF.result_path + errorFile, "\n".join(outPutList))
        print("预测完成")
        self.showPredictResult()
        self.metrics_result(testSet.label, predicted)
        print "混淆矩阵为:"
        # 显示所有列
        pd.set_option('display.max_columns', None)
        # 显示所有行
        pd.set_option('display.max_rows', None)
        print pd.DataFrame(confusion_matrix(testSet.label, predicted),
                     columns=["education", "ent", "food", "healthy", "history", "houseproperty", "military",
                              "sport", "stock", "technology"],
                     index=["education", "ent", "food", "healthy", "history", "houseproperty", "military",
                              "sport", "stock", "technology"])

    def showPredictResult(self):
        for i in range(len(self.ConfusionMatrix)):
            keyWord = ""
            for key in TF_IDF.class_code:
                if TF_IDF.class_code[key] == i:
                    keyWord = key
                    break
            print "==================================================="
            print keyWord + "类预测总数为:", np.sum([self.ConfusionMatrix[i]])
            print keyWord + "类预测正确数为:", self.ConfusionMatrix[i][i]
            for j in range(len(self.ConfusionMatrix[0])):
                if j == i:
                    continue
                predictKey = ""
                for key in TF_IDF.class_code:
                    if TF_IDF.class_code[key] == j:
                        predictKey = key
                        break
                print keyWord+"类预测为"+predictKey+"数为:", self.ConfusionMatrix[i][j]
        print "==================================================="



if __name__ == '__main__':
    TF_IDF = TF_IDF()
    # 步骤1
    # TF_IDF.loadSegmentation(TF_IDF.data_path)
    # TF_IDF.loadTestData(TF_IDF.test_path)

    # 步骤2
    # train_tfidf_path = TF_IDF.result_path+"train_tfidf.dat"
    # train_bunch_path = TF_IDF.result_path+"trainData.dat"
    # TF_IDF.calculateTFIDF(train_tfidf_path=None,bunch_path=train_bunch_path,tfidf_path=train_tfidf_path)

    # test_tfidf_path = TF_IDF.result_path+"test_tfidf.dat"
    # test_bunch_path = TF_IDF.result_path+"testData.dat"
    # TF_IDF.calculateTFIDF(train_tfidf_path=train_tfidf_path, bunch_path=test_bunch_path, tfidf_path=test_tfidf_path)

    # 步骤3
    # train_tfidf_path = TF_IDF.result_path + "train_tfidf.dat"
    # trainSet = TF_IDF.readbunchobj(train_tfidf_path)
    # TF_IDF.trainProcess(trainSet)

    # 步骤4
    test_tfidf_path = TF_IDF.result_path + "test_tfidf.dat"
    testSet = TF_IDF.readbunchobj(test_tfidf_path)
    MultinomialNBPath = TF_IDF.result_path + "MultinomialNBModel.dat"
    TF_IDF.predictProcess(MultinomialNBPath,testSet)

    # 步骤2
    # train_tfidf_path = TF_IDF.result_path + "train_tfidf_svm.dat"
    # train_bunch_path = TF_IDF.result_path + "trainData.dat"
    # TF_IDF.calculateTFIDF(train_tfidf_path=None,bunch_path=train_bunch_path,tfidf_path=train_tfidf_path)

    # test_tfidf_path = TF_IDF.result_path + "test_tfidf_svm.dat"
    # test_bunch_path = TF_IDF.result_path + "testData.dat"
    # #TF_IDF.calculateTFIDF(train_tfidf_path=train_tfidf_path, bunch_path=test_bunch_path, tfidf_path=test_tfidf_path)

    # 步骤3
    # train_tfidf_path = TF_IDF.result_path + "train_tfidf_svm.dat"
    # trainSet = TF_IDF.readbunchobj(train_tfidf_path)
    # TF_IDF.SVMProcess(trainSet)

    # 步骤4
    # test_tfidf_path = "/Users/wangyuhui/Desktop/svmResult1/test_tfidf_svm.dat"
    # testSet = TF_IDF.readbunchobj(test_tfidf_path)
    # svmPath = "/Users/wangyuhui/Desktop/forest/svmModel.dat"
    # TF_IDF.predictProcess(svmPath,testSet)

