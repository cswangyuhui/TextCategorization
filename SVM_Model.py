# -*- coding: utf-8 -*-
import os
import datetime
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
    # data_path = "/Users/wangyuhui/Desktop/dataminingData/train/"
    # test_path = "/Users/wangyuhui/Desktop/dataminingData/test/"
    # result_path = "/Users/wangyuhui/Desktop/dataminingResult1/"
    # stop_words_path = "/Users/wangyuhui/Desktop/dataminingData/stop_words_ch.txt"

    data_path = "/Users/wangyuhui/Desktop/dataminingData/train/"
    test_path = "/Users/wangyuhui/Desktop/dataminingData/test/"
    # result_path = "C:\\Users\\zo\\Desktop\\dataminingResult\\"
    result_path = "/Users/wangyuhui/Desktop/svmResult/"
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
    def loadSegmentation(self, path):
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
        self.writebunchobj(TF_IDF.result_path + "trainData.dat", bunch)
        endtime = datetime.datetime.now()
        span = endtime - begintime
        print "训练bunch:contents长度、label长度:", len(bunch.contents), len(bunch.label)
        print "训练数据保存完成,所花费时间为", span.seconds
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
            self.writebunchobj(TF_IDF.result_path+"myvocabulary.dat", vectorizer.vocabulary_)
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
        with open(TF_IDF.result_path+"svmModel.dat", "wb") as file_obj:
        # with open(TF_IDF.result_path + "svmModel.dat", "wb") as file_obj:
            pickle.dump(clf, file_obj)
        endtime = datetime.datetime.now()
        span = endtime - begintime
        print "SVM训练时间:", span.seconds

    def predictProcess(self,modelPath,testSet,predictedResult):
        begintime = datetime.datetime.now()
        if modelPath is None:
            print "模型路径未知"
            return
        clf = self.readbunchobj(modelPath)
        errorFile = "MultinomialNBerror.txt"
        if modelPath.find("svm") != -1:
            errorFile = "SVMerror.txt"
        print "加载模型成功"
        endtime = datetime.datetime.now()
        span = endtime - begintime
        begintime = endtime
        print "加载模型时间:", span.seconds
        predicted = predictedResult
        # 预测分类结果
        if predictedResult is None:
            predicted = clf.predict(testSet.tfidfmetrix)
            self.writebunchobj(TF_IDF.result_path + "predictedresult.dat", predicted)
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
                     columns=["healthy", "history", "education", "ent", "food", "houseproperty", "military",
                              "sport", "stock", "technology"],
                     index=["healthy", "history", "education", "ent", "food", "houseproperty","military","sport","stock","technology"])
        endtime = datetime.datetime.now()
        span = endtime - begintime
        print "SVM预测时间:", span.seconds

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

    def TestDemo(self,vocabulary,clf):
        healthy = "因素 肝癌 风险 乙型肝炎"
        food = "孩子 饭菜 家常菜 豆腐"
        education = "流感 步伐 流感 疑似病例 学生家长 电话 程度 流感 想象 流感 病例 人数 人 流感病毒 学生家长 情况 流感病毒 流感病毒 毒性 病毒 时期 平均温度 摄氏度 温度 病毒传播 高峰期 流感病毒 能力 流感病毒 情况 医疗 部门 疾病 控制能力 病例 病例 医疗 条件 技术 程度 留学生 流感病毒 时期 学生 签证官 学生 成功率 情况 时期 国家 情况 留学生 进程 反观 流感病毒 事件 全球 疫情 国家 全球 疑似病例 国家 病例 数 留学生 国家 情况 学生 开学 个的 事件 政府 流感病毒 流感病毒 疑似病例 患者 大学 学校 校区 学校 患者 校区 校区 病例 个体 案例 公司 公司 办理 名 学生家长 学生 猪 流感 计划 家长 公司 学生家长 计划 家长 入学 关键时刻 孩子 学期 公司 办理 世界 留学生 情况 猪 流感 常识 家长 疫情 地区 留学生 学员 传染性 疾病 常识"
        ent = "名状 海角 名单 评委 名状 图 影片 名单 电影 名状 项 赢家 海角 剧情片 男配角 新人 项 电影 海水 火焰 剧情片 男女 主角 赤壁 男配角 项 奖项 电影票房 名单 男孩 降风 原著 剧本 男孩 剧情片 女配角 演员 女主角 光芒 女星 首度 有缘 首度 移师 大使 女主角 男主角 入围者 实力 同场竞技 精彩 硬仗 演员 电影 海角 同片 局面 男孩 骗子 众人 短片 格子 群雄 歌坛 大 灌篮 主题曲 原创 电影 歌曲 电影 工作者 海角 电影票房 新台币 电影 新气象 本土 感情 代表 人物 发迹 脱俗 亮眼 国际 男星 光 电影 资深 灯光师 厂 电影 后辈 奖项 名状 共项 男主角 剧情片 视觉效果 美术设计 造型 动作 原著 剧本 原创 电影 音乐 音效 剪辑 摄影 海角 共项 剧情片 演员 男配角 原创 电影 歌曲 音效 摄影 电影 共项 剧情片 男主角 剧本 动作 视觉效果 音效 海水 火焰 共项 男主角 女主角 剧情片 美术设计 摄影 赤壁 共项 男配角 视觉效果 造型 美术设计 男孩 共项 演员 女配角 原著 剧本 剧情片 事儿 共项 男配角 剧本 造型"
        history = "小白菜 案 刑部 尚书"
        houseproperty = "外观 图 风格 印花 餐具 外观 涂鸦 花卉 图案 特色 线条 气质 厨房 格调 特质"
        military = "对岸 空军 装备 战斗机 军迷 陌生 世纪 对岸 战斗机 型号 老式 性能 盟国 天线 空空导弹 全天候 目标 吊舱 水平 机群 大陆 空军 劲敌 力量对比 大陆 空军 麾下 阵容 对岸 风光 无机 窘境 国际形势 实力 对岸 战斗机 可能性 短期内 现实 机群 对岸 量身 套装 服役 本份 对岸 空军 战斗机 时代 媒体 技术 成果 整体 性能 大陆 空军 级别 机型 服役 改进型 战斗机 单发 中型机 问世 军队 地位 小个子 空战 武器 对岸 空军 顶尖 机型 田忌赛马 对岸 马 大陆 空军 马 对岸 航电 有源 相控阵 航电 系统 技术 原型 能力 航电 功率 逊色 空战 武器 装备 头盔 显示器 基本上 射程 性能 机动性 发动机 发动机 空重 发动机 空重 航电 系统 机身 结构 优势 鸭式 布局 能力 稳盘 能力 机 后者 机体 状态 机龄 新机 起跑线 性能 力压 对岸 利器 大陆 人 事实 大度 心理 事实 作者 利刃 刻雨 无痕 内容 军事 官方 公众 军事 官方 栏目 文章 目的 信息 代表 观点 真实性 凡本网 版权所有 作品版权 作者 版权 原作者 人 本网 作者 利用 方式 作品 军事 军 军事 门户"
        sport = "爵士 胖友 距离 差分 哥 福 分差 爵士队 净输分 全场 爵士队 下半场 队史 纪录 爵士 人 队史 赢球 差 纪录 爵士 主场 专业 战 底线 手感 价格公道 问 分差 数据 小 均分 手下留情 人 数据 罚球 命中率 图 图 腿 打篮球 腿 帅气 后仰 腿 大 木桩 区别 近景 球员 镜头 印象 毛钱"
        stock = "经济 数据 财经 消息 汇率 跌势 原因 政府 报告 生产者 汇率 元 跌幅 汇率 涨幅 汇率 跌幅 报告 称份 经济学家 报告 称份 生产者 国际 货币 汇率 均告 过度 理由 货币 汇率 原因 股市 投资者 银行 证券公司 外汇 策略师 政府 报告 投资者 信心 重创 投资者 风险 情绪 跌幅 原因 抗议者 政府部门 水平 汇率 涨幅 汇率 涨幅 原因 股市 市场 投资者 利差 交易 交易 投资者 利率 国家 贷款 地区 收益 资产"
        technology = "电脑网 游戏 奖项 游戏 奖项 游戏 刺客 信条 战神 蜘蛛侠 怪物 猎人 世界 荒野 镖客 游戏 名单 玩家 荒野 镖客 荒野 镖客 荒野 镖客 游戏 言论 荒野 镖客 游戏 玩家 分量 荒野 镖客 星 游戏 游戏 历时 花费 剧情 系统 画面 游戏 机构 玩家 游戏 大作 游戏 游戏机 电视 游玩 程度 电视 画质 性能 游戏 厂商 事情 高端 电视 荒野 镖客 高端 游戏 画面 外媒 荒野 镖客 画面 事情 游戏 原生 效果 外媒 荒野 镖客 游戏 屏幕 绘制 光谱 图 记录 亮度 光谱 颜色 图像 颜色 水平 动态 颜色 基本上 图像 灰色 粉红色 内容 红色 橙色 结论 荒野 镖客 效果 玩家 选项 感觉 画面 星 玩家 效果 游戏 画质 整体 口碑 销量 游戏 笔者 朋友 朋友 游戏 玩家 游戏 总向 感觉 游戏 画质 电视 玩游戏 画质 电视 笔者 游戏 画质 游戏 画质 引擎 游戏机 机能 电视 画质 游戏 老 电视 玩游戏 游戏 画面 电视 锅 电影 同理 劲 片源 体验 锅 片源 电视 力 技术 画面 视频 技术 经历 漫长 视频 画质 分辨率 深灰 色域 空间 亮度 动态 基础 水平 时代 所幸 动态 市面上 平板电视 标准 电视 平板电视 标准 性能 产物 后者 基础 功能 静态 数据 性能 能力 门槛 公司 费用 电视 厂商 标准 创维 差异 技术 硬件 性能 企业 差距 图片 电视 影片 图 电视 影片 图 奇幻 图 整体 亮度 亮度 颜色 画面 面纱 人物 特写 差异 皮肤 颜色 大 逆光 画面 宽容度 人物 脸部皮肤 细节 细节 衣服 纹理 环境 亮度高 动态 实力 天空 云朵 逆光 飞机 暗部 细节 生机 效果 压倒性 优势 亮度 亮度 色彩 灰度 时代 想象 事物 内容 电影 游戏 电视 高标准 产品"
        content = [healthy,food,education,ent,history,houseproperty,military,sport,stock,technology]
        vectorizer = CountVectorizer(stop_words=TF_IDF.stop_words,vocabulary=vocabulary)
        content = vectorizer.fit_transform(content)
        #print content.toarray()
        label = clf.predict(content)
        print label


if __name__ == '__main__':
    TF_IDF = TF_IDF()
    # 步骤1
    # TF_IDF.loadSegmentation(TF_IDF.data_path)
    # TF_IDF.loadTestData(TF_IDF.test_path)

    # 步骤2
    # train_tfidf_path = TF_IDF.result_path + "train_tfidf_svm.dat"
    # train_bunch_path = TF_IDF.result_path + "trainData.dat"
    # TF_IDF.calculateTFIDF(train_tfidf_path=None,bunch_path=train_bunch_path,tfidf_path=train_tfidf_path)

    # test_tfidf_path = TF_IDF.result_path + "test_tfidf_svm.dat"
    # test_bunch_path = TF_IDF.result_path + "testData.dat"
    # TF_IDF.calculateTFIDF(train_tfidf_path=train_tfidf_path, bunch_path=test_bunch_path, tfidf_path=test_tfidf_path)

    # 步骤3
    # train_tfidf_path = TF_IDF.result_path + "train_tfidf_svm.dat"
    # trainSet = TF_IDF.readbunchobj(train_tfidf_path)
    # TF_IDF.SVMProcess(trainSet)

    # 步骤4
    predictedResultPath = TF_IDF.result_path + "predictedresult.dat"
    test_tfidf_path = TF_IDF.result_path + "test_tfidf_svm.dat"
    testSet = TF_IDF.readbunchobj(test_tfidf_path)
    svmPath = TF_IDF.result_path + "svmModel.dat"
    predictedResult = TF_IDF.readbunchobj(predictedResultPath)
    TF_IDF.predictProcess(svmPath,testSet,predictedResult)

    # 步骤5
    vocabulary_path = TF_IDF.result_path + "myvocabulary.dat"
    vocabulary = TF_IDF.readbunchobj(vocabulary_path)
    svmPath = TF_IDF.result_path + "svmModel.dat"
    clf = TF_IDF.readbunchobj(svmPath)
    TF_IDF.TestDemo(vocabulary,clf)


