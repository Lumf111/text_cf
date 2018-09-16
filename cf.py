import jieba
import os
import pickle
from numpy import*
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch
from sklearn.naive_bayes import MultinomialNB  #多项式贝叶斯算法

def readFile(path):
    with open (path,'r',errors='ignore') as file:  #文档中编码有问题，用errors过滤错误
        content =file.read()
        return content

def saveFile(path,result):
    with open (path,'w',errors='ignore') as file:
        file.write(result)

def segText(inputPath,resultPath):
    fatherLists=os.Listdir(inputPath)  #主目录
    for eachDir in fatherLists:        #遍历目录中各个文件夹
        eachPath=inputPath+eachDir+"/"
        each_resultPath=resultPath+eachDir+"/"
        if not os.path.exists(each_resultPath):
            os.makedirs(each_resultPath)
        childLists=os.listdir(eachPath)
        for eachFile in childLists:
            eachPathFile=eachPath + eachFile
            print (eachFile)
            content = readFile(eachPathFile)

            result=(str(content)).replace("\r\n","").strip()


            cutResult =jieba.cut(result)
            saveFile(each_resultPath+eachFile," ".join(cutResult))

def bunchSave(inputFile,outputFile):
    catelist=os.listdir(inputFile)
    bunch=Bunch(target_name=[],lable=[],filenames=[],contents=[])
    bunch.target_name.extend(catelist)
    for eachDir in catelist:
        eachPath=inputFile+eachDir+"/"
        fileList=os.listDir(eachPath)

        for eachFile in fileList:
            fullName=eachPath+eachFile
            bunch.label.append(eachDir)
            bunch.filenames.append(fullName)
            bunch.contents.append(readFile(fullName).strip())

        with open (outputFile,'wb') as file_obj:
            pickle.dump(bunch,file_obj)


def readBunch(path):
    with open (path,'rb') as file:
        bunch=pickle.loac(file)
    return bunch

def writeBunch(path,bunchFile):
    with open (path,'wb') as file:
        pickle.dump(bunchFile,file)

def getStopWord(inputFile):
    stopWordList = readFile(inputFile).splitlines()
    return stopWordList

def getTFIDFMat(inputPath,stopWordList,outputPath):
    bunch=readBunch(inputPath)
    tfidfspace=Bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})
    #初始向量化空间
    vectorizer = TfidfVectorizer(stop_words=stopWordList,sublinear_tf=True,max_df=0.5)
    transformer=TfidfTransformer()

    tfidfspace.tdm=vectorizer.fit_transform(bunch.contents)
    tfidfspace.vacabulary=vectorizer.vocabulary_
    writeBunch(outputPath,tfidfspace)

def getTestSpace(testSetPath,trainSpacePath,stopWordList,testSpacePath):
    bunch=readBunch(testSetPath)

    testSpace=bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})

    trainbunch=readBunch(trainSpacePath)

    vectorizer=TfidfVectorizer(stop_words=stopWordList,sublinear_tf=True,max_df=0.5,vocabulary=trainbunch.vocabulary)
    transformer=TfidfTransformer()
    testSpace.tdm=vectorizer.fit_transform(bunch.contents)
    testSpace.vocabulary=trainbunch.vocabulary
    #持久化
    writeBunch(testSpacePath,testSpace)

def bayesAlgorithm(trainPath,testPath):
    trainSet=readBunch(trainPath)
    testSet=readBunch(testPath)
    clf=MultinomialNB(alpha=0.001).fit(trainSet.tdm,trainSet.label)
    print(shape(trainSet.tdm))
    print(shape(testSet.tdm))
    predicted=clf.predict(testSet.tdm)
    total=len(predicted)
    rate=0
    for flabel,fileName,expct_cate in zip(testSet.label,testSet.filenames,predicted):
        if flabel!= expct_cate:
            rate+=1
            print(fileName, ":实际类别：", flabel, "-->预测类别：", expct_cate)

    print("error rate:",float(rate)*100/float(total),"%")

    #分词，第一个是分词输入，第二个参数是结果保存的路径
    segText("D:/develop/python/text_cf/Train_Data/train_corpus/","D:/develop/python/Train_Data/train_corpus_seg/")
    bunchSave("D:/Train_Data/segResult","D:/Train_Data/Train_set.dat")
    stopWordList=getStopWord("D:/Train_Data/stopwords.txt")
    getTFIDFMat("D:/Train_Data/Train_set.dat",stopWordList,"D:/Train_data/tfidfspace.dat")


    #训练集
    segText("D:/develop/python/Train_Data/test_corpus","D:/Train_Data/test_corpus_seg")
    bunchSave("D:Train_Data/test_segResult","D:/Train_Data/test_set.dat")
    getTestSpace("D:/Train_Data/test_set.dat","D:/Train_Data/tfidfspace.dat",stopWordList,"D:/Train_Data/testspace.dat")
    bayesAlgorithm("D:/Train_Data/tfidfspave.dat","D/Train_Data/testspace.dat")
