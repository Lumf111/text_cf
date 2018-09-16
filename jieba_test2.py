import jieba
def readFile(path):
    with open(path,'r',errors='ignore') as file:
        content=file.read()
        return content

def saveFile(path,result,errors='ignore'):
    with open(path,'w') as file:
        file.write(result)


content=readFile("D:/develop/python/text_cf/test.txt")
result=(str(content)).replace("\r\n","").strip() #删除多余空格与空行
cutResult=jieba.cut(result)#默认方式分词，分词结果用空格隔开
saveFile("D:/develop/python/text_cf/result.txt"," ".join(cutResult))



