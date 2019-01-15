# TextCategorization
北邮王晓茹老师数据挖掘与数据仓库文本数据的分类与分析实验

## 程序结构

|程序|功能|
|:---|:---|
|finance.py|爬取新闻的链接|
|read_finance.py|根据新闻链接访问新闻,并将新闻写到数据库中|
|WordSegmentation.py|数据清洗、去除停用词、分词|
|TF_IDF_Model.py|朴素贝叶斯算法|
|SVM_Model.py|svm算法|

## 程序结果
### 朴素贝叶斯结果

<div align=center><img width="480" height="360" src="https://github.com/cswangyuhui/TextCategorization/blob/master/result/bayes_result.png"/></div>
<br>
<div align=center><img width="480" height="360" src="https://github.com/cswangyuhui/TextCategorization/blob/master/result/bayes_confusion_matrix.png"/></div>

### svm结果
<div align=center><img width="480" height="360" src="https://github.com/cswangyuhui/TextCategorization/blob/master/result/svm_result.png"/></div>
<br>
<div align=center><img width="480" height="360" src="https://github.com/cswangyuhui/TextCategorization/blob/master/result/svm_confusion_matrix.png"/></div>
