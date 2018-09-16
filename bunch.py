import os
import pickle
from sklearn.datasets.base import Bunch

from Tools import readfile
def corpus2Bunch(wordbag_path, seg_path):

    catelist = os.listdir(seg_path)  # 获取seg_path下的所有子目录，也就是分类信息

    # 创建一个Bunch实例

    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])

    bunch.target_name.extend(catelist)

    # 获取每个目录下所有的文件

    for mydir in catelist:

        class_path = seg_path + mydir + "/"  # 拼出分类子目录的路径

        file_list = os.listdir(class_path)  # 获取class_path下的所有文件

        for file_path in file_list:  # 遍历类别目录下文件

            fullname = class_path + file_path  # 拼出文件名全路径

            bunch.label.append(mydir)

            bunch.filenames.append(fullname)

            bunch.contents.append(readfile(fullname))  # 读取文件内容

    # 将bunch存储到路径中

    with open(wordbag_path, "wb") as file_obj:

        pickle.dump(bunch, file_obj)

    print("构建文本对象结束！！！")


if __name__ == "__main__":

    # 对训练集进行Bunch化操作：

    wordbag_path = "D:/develop/python/text_cf/Train_Data/train_set.dat"  # Bunch存储路径

    seg_path = "D:/develop/python/text_cf/Train_Data/train_corpus_seg/"  # 分词后分类语料库路径

    corpus2Bunch(wordbag_path, seg_path)

    # 对测试集进行Bunch化操作：

    wordbag_path = "D:/develop/python/text_cf/Train_Data/test_set.dat"  # Bunch存储路径

    seg_path = "D:/develop/python/text_cf/Train_Data/test_corpus_seg/"  # 分词后分类语料库路径

    corpus2Bunch(wordbag_path, seg_path)