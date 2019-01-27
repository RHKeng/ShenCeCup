#!/usr/bin/env python
# -*-coding:utf-8-*-

'''

'''

import struct
import os
import pandas as pd
import numpy as np
from pyltp import SentenceSplitter
import functools
from pyltp import Segmentor
from pyltp import Postagger
import jieba
import jieba.posseg as psg
from jieba import analyse
import re
import math

# 原始字节码转为字符串
def byte2str(data):
    pos = 0
    str = ''
    while pos < len(data):
        c = chr(struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0])
        if c != chr(0):
            str += c
        pos += 2
    return str

# 获取拼音表
def getPyTable(data, GPy_Table, GTable):
    data = data[4:]
    pos = 0
    while pos < len(data):
        index = struct.unpack('H', bytes([data[pos],data[pos + 1]]))[0]
        pos += 2
        lenPy = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
        pos += 2
        py = byte2str(data[pos:pos + lenPy])

        GPy_Table[index] = py
        pos += lenPy
    return GPy_Table, GTable

# 获取一个词组的拼音
def getWordPy(data, GPy_Table, GTable):
    pos = 0
    ret = ''
    while pos < len(data):
        index = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
        ret += GPy_Table[index]
        pos += 2
    return ret, GPy_Table, GTable

# 读取中文表
def getChinese(data, GPy_Table, GTable):
    pos = 0
    while pos < len(data):
        # 同音词数量
        same = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]

        # 拼音索引表长度
        pos += 2
        py_table_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]

        # 拼音索引表
        pos += 2
        py, GPy_Table, GTable = getWordPy(data[pos: pos + py_table_len], GPy_Table, GTable)

        # 中文词组
        pos += py_table_len
        for i in range(same):
            # 中文词组长度
            c_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
            # 中文词组
            pos += 2
            word = byte2str(data[pos: pos + c_len])
            # 扩展数据长度
            pos += c_len
            ext_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
            # 词频
            pos += 2
            count = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]

            # 保存
            GTable.append((count, py, word))

            # 到下个词的偏移位置
            pos += ext_len
    return GPy_Table, GTable

def scel2txt(file_name, startPy, startChinese, GPy_Table, GTable):
    print('-' * 60)
    with open(file_name, 'rb') as f:
        data = f.read()

    # print("词库名：", byte2str(data[0x130:0x338])) # .encode('GB18030')
    # print("词库类型：", byte2str(data[0x338:0x540]))
    # print("描述信息：", byte2str(data[0x540:0xd40]))
    # print("词库示例：", byte2str(data[0xd40:startPy]))
    GPy_Table, GTable = getPyTable(data[startPy:startChinese], GPy_Table, GTable)
    GPy_Table, GTable = getChinese(data[startChinese:], GPy_Table, GTable)
    return GPy_Table, GTable

#停用词表加载方法
def get_stopword_list():
    #停用词表存储路径，每一行为一个词，按行读取进行加载
    #进行编码转换确保匹配准确率
    stop_word_path = './stopword.txt'
    stop_word_list = [sw.replace('\n', '') for sw in open(stop_word_path).readlines()]
    return stop_word_list

#分词方法，调用结巴接口
def jieba_seg_to_list(sentence, pos=False):
    if not pos:
        #不进行词性标注的分词方法
        seg_list = jieba.cut(sentence)
    else:
        #进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    return seg_list

#去除干扰词
def jieba_word_filter(seg_list, stopword_list, pos=False):

    filter_list = []
    #根据pos参数选择是否词性过滤
    #不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        #过滤高停用词表中的词，以及长度为<2的词
        if not word in stopword_list and len(word) > 1:
            filter_list.append(word)

    return filter_list

def jieba_word_deal(sentence, stopword_list, pos=False):
    #调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
    seg_list = jieba_seg_to_list(sentence, pos)
    filter_list = jieba_word_filter(seg_list, stopword_list, pos)
    return filter_list

def jieba_title_word_n(sentence, stopword_list):
    #调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
    seg_list = jieba_seg_to_list(sentence, True)
    title_word_n_list = []
    for seg in seg_list:
        word = seg.word
        flag = seg.flag
        if flag.startswith('n'):
            title_word_n_list.append(word)
    return title_word_n_list

#分词方法，调用ltp接口
def ltp_seg_to_list(sentence, segmentor):
    words = segmentor.segment(sentence)  # 分词
    seg_list = list(words)
    return seg_list

#去除干扰词
def ltp_word_filter(seg_list, stopword_list):

    filter_list = []
    #根据pos参数选择是否词性过滤
    #不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        #过滤高停用词表中的词，以及长度为<2的词
        if not seg in stopword_list and len(seg) > 1:
            filter_list.append(seg)

    return filter_list

def ltp_word_deal(sentence, stopword_list, segmentor):
    #调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
    seg_list = ltp_seg_to_list(sentence, segmentor)
    filter_list = ltp_word_filter(seg_list, stopword_list)
    return filter_list

def get_title_person_name(title_list, postagger):
    attributes_list = list(postagger.postag(title_list))
    title_person_name_list = list()
    i = 0
    for attributes in attributes_list:
        if attributes == 'nh':
            title_person_name_list.append(title_list[i])
        i = i + 1
    return title_person_name_list

def get_text_sentences(text):
    sentences_list = SentenceSplitter.split(text)
    return sentences_list

def get_title_text(all_docs_df):
    temp_df = pd.DataFrame(columns=['id', 'title_text'])
    for temp_id, title, text, text_sentences_len in all_docs_df[['id', 'title', 'text', 'text_sentences_len']].values:
        length = math.ceil(text_sentences_len * 0.4)
        if length < 6:
            length = 6
#         if length > 15:
#             length = 15
        title_text = ''
        for i in range(length):
            title_text = title + '。' + title_text
        title_text = title_text + text
        temp = pd.DataFrame([[temp_id, title_text]], columns=['id', 'title_text'])
        temp_df = pd.concat([temp_df, temp])
    all_docs_df = pd.merge(all_docs_df, temp_df, on='id', how='left')
    return all_docs_df

#idf统计方法
def train_idf(doc_list):
    idf_dic = {}
    #总文档数
    tt_count = len(doc_list)

    #每个词出现的文档数
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0

    #按公式转换为idf值，分母加1进行平滑处理
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(tt_count / (1.0 + v))

    #对于没有在字典中的词，默认其尽在一个文档中出现，得到默认idf值
    default_idf = math.log(tt_count / (1.0))
    return idf_dic, default_idf

#排序函数，用于topK关键词的按值排
def cmp(e1, e2):
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1

#TF-IDF类
class TfIdf(object):
    #四个参数分别是：训练好的idf字典，默认idf值，处理后的待提取文本，关键词数量
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dic = idf_dic
        self.default_idf = default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num

    #统计tf值
    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0

        tt_count = len(self.word_list)
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count

        return tf_dic

    #按公式计算tf-idf
    def get_tfidf(self):
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)

            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        result_dict = {}
        for k, v in sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            result_dict[k] = result_dict.get(k, 0.0) + float(v)
        return result_dict

def is_contain_alpha(x):
    my_re = re.compile(r'[A-Za-z]',re.S)
    res = re.findall(my_re,x)
    if len(res):
        return True
    else:
        return False

def get_deal_result_list(keyword_set, sample_df):
    temp_df = pd.DataFrame(columns=['id', 'result_list'])
    for temp_id, jieba_title_list, result_dict, jieba_title_person_name_list, ltp_title_person_name_list, title, jieba_title_word_n_list in sample_df[['id', 'jieba_title_list', 'jieba_result_dict_10', 'jieba_title_person_name_list', 'ltp_title_person_name_list', 'title', 'jieba_title_word_n_list']].values:
        show_list = list()
        word_list = re.findall(r"《(.+?)》", title)
        for word in word_list:
            if ',' in word:
                word = word.replace(',', '，')
                print(word)
            show_list.append(word)
        title_set = set(jieba_title_list)
        keys = list(result_dict.keys())
        title_person_name_set = set(jieba_title_person_name_list) & set(ltp_title_person_name_list)
#         title_person_name_set = set(jieba_title_person_name_list)
        jieba_title_word_n_set = set(jieba_title_word_n_list)
        result_list = list()
        result_list_1 = list()
        result_list_2 = list()
        result_list_3 = list()
        result_list_4 = list()
        result_list_5 = list()
        result_list_6 = list()
        result_list_7 = list()
        for title in title_set:
            if len(title) >= 5:
                result_list.append(title)
        for key in keys:
            if key in set(result_list):
                continue
            if((key in keyword_set) & (key in title_set)):
                result_list_1.append(key)
            else:
                if key in title_person_name_set:
                    result_list_5.append(key)
                else:
                    if (((key in set(jieba_title_person_name_list)) | (key in set(ltp_title_person_name_list))) & (len(key) >= 3)):
                        continue
                    else:
#                         if is_contain_alpha(key):
#                             result_list_7.append(key)
#                         else:
                        if key in title_set:
                            if key in jieba_title_word_n_set:
                                result_list_7.append(key)
                            else:
                                result_list_3.append(key)
                        else:
                            if key in keyword_set:
                                result_list_2.append(key)
                            else:
                                result_list_4.append(key)
        for name in set(jieba_title_person_name_list):
            if name in set(result_list_5):
                continue
            else:
                if len(name) >= 3:
                    result_list_6.append(name)
        result_list = show_list + result_list_5 + result_list_1 + result_list + result_list_6 + result_list_7 + result_list_3 + result_list_2 + result_list_4
        final_list = list()
        for result in result_list:
            if result not in final_list:
                final_list.append(result)
        temp = pd.DataFrame([[temp_id, final_list]], columns=['id', 'result_list'])
        temp_df = pd.concat([temp_df, temp])
    print(temp_df.head())
    sample_df = pd.merge(sample_df, temp_df, on='id', how='left')
    return sample_df

def get_top_n_word(result_list, n):
    if len(result_list) < n:
        return '无'
    else:
        return result_list[n - 1]

# 导出预测结果
def exportResult(df, fileName):
    df.to_csv('./%s.csv' % fileName, header=True, index=False)

def main():

    # 词典处理
    print('~~~~~~~~~~~~~~开始处理词典~~~~~~~~~~~~~~~~~~~')
    # 拼音表偏移，
    startPy = 0x1540;

    # 汉语词组表偏移
    startChinese = 0x2628;

    # 全局拼音表
    GPy_Table = {}

    # 解析结果
    # 元组(词频,拼音,中文词组)的列表
    GTable = []

    # scel所在文件夹路径
    in_path = "./dict/"
    # 输出词典所在文件夹路径
    out_path = "./user_dict.txt"

    fin = [fname for fname in os.listdir(in_path) if fname[-5:] == ".scel"]
    for f in fin:
        f = os.path.join(in_path, f)
        GPy_Table, GTable = scel2txt(f, startPy, startChinese, GPy_Table, GTable)

    # 保存结果
    with open(out_path, 'w', encoding='utf8') as f:
        f.writelines([word+'\n' for count, py, word in GTable])

    print('~~~~~~~~~~~~~~词典处理完毕~~~~~~~~~~~~~~~~~~~')

    all_docs_df = pd.read_csv('./all_docs.txt', sep='\001', header=None)
    all_docs_df.columns = ['id', 'title', 'text']
    all_docs_df['title'] = all_docs_df['title'].astype(str)
    all_docs_df['text'] = all_docs_df['text'].astype(str)

    train_doc_keyword_df = pd.read_csv('./train_docs_keywords.txt', sep='\t', header=None)
    train_doc_keyword_df.columns = ['id', 'keyword']
    train_doc_keyword_df['keyword_list'] = train_doc_keyword_df['keyword'].map(lambda x: x.split(','))
    jieba.load_userdict('user_dict.txt')
    #给jieba添加自定义词
    for keyword_list in train_doc_keyword_df['keyword_list']:
        for keyword in keyword_list:
            jieba.add_word(keyword)
    keyword_set = set()
    for keyword_list in train_doc_keyword_df['keyword_list']:
        for keyword in keyword_list:
            keyword_set.add(keyword)
    LTP_DATA_DIR = './ltp_data_v3.4.0/'  # ltp模型目录的路径
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型

    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    postagger = Postagger() # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    print('~~~~~~~~~~~~~~开始进行分词~~~~~~~~~~~~~~~~~~~')
    all_docs_df['text_sentences'] = all_docs_df['text'].map(lambda x: get_text_sentences(x))
    all_docs_df['text_sentences_len'] = all_docs_df['text_sentences'].map(lambda x: len(x))

    stopword_list = get_stopword_list()
    all_docs_df['jieba_title_list'] = all_docs_df['title'].map(lambda x : jieba_word_deal(x, stopword_list, False))
    all_docs_df['jieba_title_word_n_list'] = all_docs_df['title'].map(lambda x : jieba_title_word_n(x, stopword_list))
    all_docs_df = get_title_text(all_docs_df)
    all_docs_df['jieba_title_text_list'] = all_docs_df['title_text'].map(lambda x : jieba_word_deal(x, stopword_list, False))

    all_docs_df['ltp_title_list'] = all_docs_df['title'].map(lambda x : ltp_word_deal(x, stopword_list, segmentor))
    all_docs_df['ltp_title_text_list'] = all_docs_df['title_text'].map(lambda x : ltp_word_deal(x, stopword_list, segmentor))

    all_docs_df['jieba_title_person_name_list'] = all_docs_df['jieba_title_list'].map(lambda x: get_title_person_name(x, postagger))
    all_docs_df['ltp_title_person_name_list'] = all_docs_df['ltp_title_list'].map(lambda x: get_title_person_name(x, postagger))
    print('~~~~~~~~~~~~~~分词完毕~~~~~~~~~~~~~~~~~~~')

    print('~~~~~~~~~~~~~~开始进行tfidf统计~~~~~~~~~~~~~~~~~~~')
    jieba_idf_dic, jieba_default_idf = train_idf(all_docs_df['jieba_title_text_list'])
    all_docs_df['jieba_result_dict_5'] = all_docs_df['jieba_title_text_list'].map(lambda x: TfIdf(jieba_idf_dic, jieba_default_idf, x, 5).get_tfidf())
    all_docs_df['jieba_result_dict_10'] = all_docs_df['jieba_title_text_list'].map(lambda x: TfIdf(jieba_idf_dic, jieba_default_idf, x, 10).get_tfidf())
    ltp_idf_dic, ltp_default_idf = train_idf(all_docs_df['ltp_title_text_list'])
    all_docs_df['ltp_result_dict_5'] = all_docs_df['ltp_title_text_list'].map(lambda x: TfIdf(ltp_idf_dic, ltp_default_idf, x, 5).get_tfidf())
    all_docs_df['ltp_result_dict_10'] = all_docs_df['ltp_title_text_list'].map(lambda x: TfIdf(ltp_idf_dic, ltp_default_idf, x, 10).get_tfidf())
    print('~~~~~~~~~~~~~~tfidf统计完毕~~~~~~~~~~~~~~~~~~~')

    sample_df = pd.read_csv('./sample.csv', encoding='ISO-8859-1')
    sample_df = pd.merge(sample_df, all_docs_df, on='id', how='left')

    print('~~~~~~~~~~~~~~开始进行规则处理~~~~~~~~~~~~~~~~~~~')
    sample_df = get_deal_result_list(keyword_set, sample_df)
    sample_df['label1'] = sample_df['result_list'].map(lambda x: get_top_n_word(x, 1))
    sample_df['label2'] = sample_df['result_list'].map(lambda x: get_top_n_word(x, 2))
    print('~~~~~~~~~~~~~~规则处理完毕~~~~~~~~~~~~~~~~~~~')

    exportResult(sample_df[['id', 'label1', 'label2']], 'tfidf_final')
    print('~~~~~~~~~~~~~~完毕~~~~~~~~~~~~~~~~~~~')


if __name__ == '__main__':
    main()
