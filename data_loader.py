import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import transformers
from transformers import AutoTokenizer
from os.path import join

from config import *

# DATA_DIR = './dataset/eca_ch'
# TRAIN_FILE = 'fold%s_train.txt'
# TEST_FILE = 'fold%s_test.txt'
# batch_size = 2
# epochs = 15
# bert_cache_path = './bert_base_ch'

# 阿拉伯数字转汉字
_MAPPING = (
    u'零', u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'十', u'十一', u'十二', u'十三', u'十四',
    u'十五', u'十六', u'十七', u'十八', u'十九')
_P0 = (u'', u'十', u'百', u'千',)
_S4 = 10 ** 4


def num_to_chinese4(num):
    assert (0 <= num and num < _S4)
    if num < 20:
        return _MAPPING[num]
    else:
        lst = []
        while num >= 10:
            lst.append(num % 10)
            num = num / 10
        lst.append(num)
        c = len(lst)  # 位数
        result = u''

        for idx, val in enumerate(lst):
            val = int(val)
            if val != 0:
                result += _P0[idx] + _MAPPING[val]
                if idx < c - 1 and lst[idx + 1] == 0:
                    result += u'零'
        return result[::-1]

def get_emotion_num_list(configs, data_file):
    data_list = load_txt_2_list(data_file, configs.total_prompt)
    emotion_num_list = [0] * 7
    for doc in data_list:
        doc_clauses = doc['clauses']
        doc_len = doc['doc_len']
        for i in range(doc_len):
            clause = doc_clauses[i]
            # emotion_label = int(i + 1 in doc_emotions)
            # 添加情感标签
            if clause['emotion_category'] == 'happiness':
                emotion_num_list[1] += 1
            elif clause['emotion_category'] == 'sadness':
                emotion_num_list[2] += 1
            elif clause['emotion_category'] == 'fear':
                emotion_num_list[3] += 1
            elif clause['emotion_category'] == 'anger':
                emotion_num_list[4] += 1
            elif clause['emotion_category'] == 'disgust':
                emotion_num_list[5] += 1
            elif clause['emotion_category'] == 'surprise':
                emotion_num_list[6] += 1
            else:
                emotion_num_list[0] += 1
    return np.array(emotion_num_list)



def load_txt_2_list(data_file, total_prompt):
    inputFile = open(data_file, 'r', encoding="utf-8")
    all_data = []
    while True:
        doc_i_dict = dict()
        line = inputFile.readline()
        if line == '':
            break
        # 添加每段文本的id和长度
        line = line.strip().split()
        doc_id = line[0]
        doc_i_dict['doc_id'] = doc_id
        doc_len = int(line[1])
        doc_i_dict['doc_len'] = doc_len

        # 添加情感句和原因句对
        pairs = eval('[' + inputFile.readline().strip() + ']')
        e_list, c_list = [], []
        for ecp in pairs:
            if ecp[0] not in e_list:
                e_list.append(ecp[0]-1)
            if ecp[1] not in c_list:
                c_list.append(ecp[1]-1)
        doc_i_dict['e_list'] = e_list
        doc_i_dict['ecps'] = pairs
        # print(doc_i_dict['ecps'])
#         doc_lines.append(doc_i_dict)

        # 添加文本的每个句子
        # 创建带prompt的句子
        doc_lines = []
        for i in range(doc_len):
            dic = dict()
            dic['clause_id'] = i + 1
#             if i + 1 in e_list:
#                 dic['emotion_category'] = 1
#             else:
#                 dic['emotion_category'] = 0
            if i + 1 in c_list:
                dic['cause_category'] = 1
            else:
                dic['cause_category'] = 0
            clause_line = inputFile.readline().strip().split(',')
            emotion_category = clause_line[1]
            emotion_token = clause_line[2]
            contents = (clause_line[-1]).replace(' ', '')
            dic['emotion_category'] = emotion_category
            dic['emotion_token'] = emotion_token
            dic['clause'] = contents

            # 添加prompt的文本
            dic['clause_prompt'] = contents
            for sub_prompt in total_prompt:
                dic['clause_prompt'] = dic['clause_prompt'] + '，' + total_prompt[sub_prompt]['prompt']
                # print("sub_prompt:", sub_prompt)
            dic['clause_prompt'] = num_to_chinese4(dic['clause_id']) + '、' + dic['clause_prompt'] + '。'
            # print("clause_prompt", dic['clause_prompt'])
            doc_lines.append(dic)
        doc_i_dict['clauses'] = doc_lines
        # print(doc_lines)
        all_data.append(doc_i_dict)
        # print('all_data:', all_data)
    return all_data


def create_train_data_loader(configs, fold_id, shuffle=True):
    train_dataset = MyDataset(configs, fold_id, data_type='train')
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                                    shuffle=shuffle, collate_fn=bert_batch_preprocessing)
    return train_data_loader


def create_test_data_loader(configs, fold_id, data_type):
    test_dataset = MyDataset(configs, fold_id, data_type)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                                   shuffle=False, collate_fn=bert_batch_preprocessing)
    return test_data_loader

def create_sampler_data_loader(configs, data_type):
    sampler_dataset = MyDataset(configs, fold_id=0, data_type=data_type)
    sampler_data_loader = torch.utils.data.DataLoader(dataset=sampler_dataset, batch_size=configs.batch_size,
                                                      shuffle=True, collate_fn=bert_batch_preprocessing)
    return sampler_data_loader


class MyDataset(Dataset):
    def __init__(self, configs, fold_id, data_type, data_dir=DATA_DIR):

        self.data_dir = data_dir
        self.data_type = data_type
        self.train_file = join(data_dir, TRAIN_FILE % fold_id)
        # print("self.train_file", self.train_file)
        # self.valid_file = join(data_dir, VALID_FILE % fold_id)
        self.test_file = join(data_dir, TEST_FILE % fold_id)
        # print("self.test_file", self.test_file)
        self.sampler_file = join(data_dir, SAMPLER_FILE)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.prompt = configs.total_prompt    # 取出prompt

        self.bert_tokenizer = AutoTokenizer.from_pretrained(configs.bert_cache_path)

        self.doc_couples_list, self.y_emo_id_list, self.y_emotions_list, self.y_causes_list, self.y_pairs_list, \
        self.doc_len_list, self.doc_id_list, \
        self.bert_token_idx_list, self.bert_clause_idx_list, self.bert_segments_idx_list, \
        self.bert_token_lens_list, \
        self.bert_token_idx_prompt_list, self.bert_clause_idx_CLS_list, self.bert_clause_idx_MASK_list, self.bert_segments_idx_prompt_list, \
        self.bert_token_lens_prompt_list = self.read_data_file(self.data_type)

    def __len__(self):
        return len(self.y_emotions_list)

    def __getitem__(self, idx):
        doc_couples, y_emo_id, y_emotions, y_causes, y_pairs = self.doc_couples_list[idx], self.y_emo_id_list[idx], self.y_emotions_list[idx], self.y_causes_list[idx], self.y_pairs_list[idx]
        doc_len, doc_id = self.doc_len_list[idx], self.doc_id_list[idx]
        bert_token_idx, bert_clause_idx = self.bert_token_idx_list[idx], self.bert_clause_idx_list[idx]
        bert_segments_idx, bert_token_lens = self.bert_segments_idx_list[idx], self.bert_token_lens_list[idx]
        bert_token_idx_prompt, bert_clause_idx_prompt_MASK = self.bert_token_idx_prompt_list[idx], self.bert_clause_idx_MASK_list[idx]
        bert_clause_idx_prompt_CLS = self.bert_clause_idx_CLS_list[idx]
        bert_segments_idx_prompt, bert_token_lens_prompt = self.bert_segments_idx_prompt_list[idx], self.bert_token_lens_prompt_list[idx]

        assert len(bert_clause_idx_prompt_CLS) == doc_len

        if bert_token_lens > 512 or bert_token_lens_prompt > 512:
            bert_token_idx, bert_clause_idx, \
            bert_segments_idx, bert_token_lens, \
            bert_token_idx_prompt, bert_clause_idx_prompt_MASK, \
            bert_clause_idx_prompt_CLS, \
            bert_segments_idx_prompt, bert_token_lens_prompt, \
            doc_couples, y_emo_id, y_emotions, y_causes, y_pairs, doc_len = self.token_trunk(bert_token_idx, bert_clause_idx,
                                                                          bert_segments_idx, bert_token_lens,
                                                                          bert_token_idx_prompt, bert_clause_idx_prompt_MASK,
                                                                          bert_clause_idx_prompt_CLS,
                                                                          bert_segments_idx_prompt, bert_token_lens_prompt,
                                                                          doc_couples, y_emo_id, y_emotions, y_causes, y_pairs, doc_len)

        # 将y_pairs中与删除的句子有关的句子更新
        for i in range(len(y_pairs)):
            if y_pairs[i] > doc_len:
                y_pairs[i] = 0

        # 调整y_emo_id，如果y_emo_id大于句子数则将其为-1
        for i in range(len(y_emo_id)):
            if y_emo_id[i] > doc_len:
                del y_emo_id[i]
                i -= 1

        bert_token_idx = torch.LongTensor(bert_token_idx)
        bert_segments_idx = torch.LongTensor(bert_segments_idx)
        bert_clause_idx = torch.LongTensor(bert_clause_idx)

        bert_token_idx_prompt = torch.LongTensor(bert_token_idx_prompt)
        bert_segments_idx_prompt = torch.LongTensor(bert_segments_idx_prompt)
        bert_clause_idx_prompt_MASK = torch.LongTensor(bert_clause_idx_prompt_MASK)

        bert_token_lens = len(bert_token_idx)
        bert_token_lens_prompt = len(bert_token_idx_prompt)

        # print('y_emotions:', y_emotions)
        # print('y_causes:', y_causes)
        # print("y_pairs", y_pairs)
        # print("bert_clause_idx_prompt_MASK_list", bert_clause_idx_prompt_MASK)

        assert doc_len == len(y_emotions)
        return doc_couples, y_emo_id, y_emotions, y_causes, y_pairs, doc_len, doc_id, \
               bert_token_idx, bert_segments_idx, bert_clause_idx, bert_token_lens, \
               bert_token_idx_prompt, bert_segments_idx_prompt, bert_clause_idx_prompt_MASK, bert_token_lens_prompt

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file
        elif data_type == 'sampler':
            data_file = self.sampler_file

        doc_id_list = []
        doc_len_list = []
        doc_couples_list = []

        y_emotions_list, y_causes_list = [], []
        y_emo_id_list = []
        y_pairs_list = []

        bert_token_idx_list = []
        bert_clause_idx_list = []
        bert_segments_idx_list = []
        bert_token_lens_list = []

        bert_token_idx_prompt_list = []
        bert_clause_idx_prompt_CLS_list = []
        bert_clause_idx_prompt_MASK_list = []
        bert_segments_idx_prompt_list = []
        bert_token_lens_prompt_list = []

        data_list = load_txt_2_list(data_file, self.prompt)
        # print('data_list:', data_list)
        for doc in data_list:
            # print('doc:', doc)
            doc_id = doc['doc_id']
            doc_len = doc['doc_len']
            doc_e_list = doc['e_list']
            doc_couples = doc['ecps']
            doc_emotions, doc_causes = zip(*doc_couples)    # 表示解压，将doc_couples的情感原因句子对分开成相应情感和原因句子
            doc_id_list.append(doc_id)
            doc_len_list.append(doc_len)
            doc_couples = list(map(lambda x: list(x), doc_couples))
            # print(doc_couples)
            doc_couples_list.append(doc_couples)
            # print(doc_id, "doc_couples_list", doc_couples_list)
            # print('doc_couples_list：', doc_couples_list)

            y_emotions, y_causes = [], [] ##labels, emotion/cause clause is 1
            doc_clauses = doc['clauses']
            doc_str = ''
            doc_str_prompt = ''
            for i in range(doc_len):

                clause = doc_clauses[i]
                # emotion_label = int(i + 1 in doc_emotions)
                # 添加情感标签
                if clause['emotion_category'] == 'happiness':
                    emotion_label = 1
                elif clause['emotion_category'] == 'sadness':
                    emotion_label = 2
                elif clause['emotion_category'] == 'fear':
                    emotion_label = 3
                elif clause['emotion_category'] == 'anger':
                    emotion_label = 4
                elif clause['emotion_category'] == 'disgust':
                    emotion_label = 5
                elif clause['emotion_category'] == 'surprise':
                    emotion_label = 6
                else:
                    emotion_label = 0

                cause_label = int(i + 1 in doc_causes)
                # print("emotion_label", emotion_label)
                # print("cause_label", cause_label)
                y_emotions.append(emotion_label)
                y_causes.append(cause_label)


                clause_id = clause['clause_id']
                assert int(clause_id) == i + 1
                doc_str += '[CLS] ' + clause['clause'] + ' [SEP] '
                doc_str_prompt += '[CLS] ' + clause['clause_prompt'] + ' [SEP] '    # prompt添加标签
            # print('y_emotions:', y_emotions)
            # print('y_causes:', y_causes)
            # print('doc_str:', doc_str)
            # print('doc_str_prompt:', doc_str_prompt)

            # 构建情感原因对的序列
            y_pairs = [0] * doc_len
            for pair in doc_couples:
                y_pairs[pair[0]-1] = pair[1]
                y_pairs[pair[1]-1] = pair[0]
            # print("y_pairs", y_pairs)

            indexed_tokens = self.bert_tokenizer.encode(doc_str.strip(), add_special_tokens=False)
            indexed_tokens_prompt = self.bert_tokenizer.encode(doc_str_prompt.strip(), add_special_tokens=False)    # prompt句子编码
            # print('indexed_tokens:', indexed_tokens_prompt)

            clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 101] ## tokens 中cls的index
            clause_indices_prompt_CLS = [i for i, x in enumerate(indexed_tokens_prompt) if x == 101]    # prompt中CLS的位置
            clause_indices_prompt_MASK = [i for i, x in enumerate(indexed_tokens_prompt) if x == 103]   # 寻找[MASK]的index

            # print('clause_indices:', clause_indices_prompt)
            doc_token_len = len(indexed_tokens)
            doc_token_len_prompt = len(indexed_tokens_prompt)

            segments_ids = self.get_segment_ids(indexed_tokens)
            segments_ids_prompt = self.get_segment_ids(indexed_tokens_prompt)
            # assert segments_ids_test == segments_ids
            # assert segments_ids_prompt_test == segments_ids_prompt

            assert len(clause_indices) == doc_len
            assert len(segments_ids) == len(indexed_tokens)
            assert len(clause_indices_prompt_CLS) == doc_len
            assert len(clause_indices_prompt_MASK) == doc_len * len(self.prompt)
            assert len(segments_ids_prompt) == len(indexed_tokens_prompt)

            bert_token_idx_list.append(indexed_tokens)
            bert_clause_idx_list.append(clause_indices)
            bert_segments_idx_list.append(segments_ids)
            bert_token_lens_list.append(doc_token_len)

            bert_token_idx_prompt_list.append(indexed_tokens_prompt)    # 加prompt的句子的id序列
            bert_clause_idx_prompt_CLS_list.append(clause_indices_prompt_CLS)   # 加prompt的句子的[CLS]的索引序列
            bert_clause_idx_prompt_MASK_list.append(clause_indices_prompt_MASK)   # 加prompt的句子的[MASK]的索引序列
            bert_segments_idx_prompt_list.append(segments_ids_prompt)   # 加prompt的句子的segment的序列
            bert_token_lens_prompt_list.append(doc_token_len_prompt)           # 句子数

            y_emotions_list.append(y_emotions)
            y_causes_list.append(y_causes)
            y_pairs_list.append(y_pairs)
            y_emo_id_list.append(doc_e_list)



        return doc_couples_list, y_emo_id_list, y_emotions_list, y_causes_list, y_pairs_list, doc_len_list, doc_id_list, \
               bert_token_idx_list, bert_clause_idx_list, bert_segments_idx_list, bert_token_lens_list, \
                bert_token_idx_prompt_list, bert_clause_idx_prompt_CLS_list, bert_clause_idx_prompt_MASK_list, bert_segments_idx_prompt_list, bert_token_lens_prompt_list

    # 计算序列的segment
    def get_segment_ids(self, indexed_tokens):
        segments_ids = []
        segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
        segments_indices.append(len(indexed_tokens))
        # print("segments_indices:", segments_indices)
        for i in range(len(segments_indices) - 1):
            semgent_len = segments_indices[i + 1] - segments_indices[i]
            if i % 2 == 0:
                segments_ids.extend([0] * semgent_len)
            else:
                segments_ids.extend([1] * semgent_len)
        return segments_ids

    def token_trunk(self, bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens,
                    bert_token_idx_prompt, bert_clause_idx_prompt, bert_clause_idx_prompt_CLS, bert_segments_idx_prompt, bert_token_lens_prompt,
                    doc_couples, y_emo_id, y_emotions, y_causes, y_pairs, doc_len):
        # TODO: cannot handle some extreme cases now
        emotion, cause = doc_couples[0]
        if emotion >= doc_len / 2 or cause >= doc_len / 2:
            i = 0
            j = i
            while True:
                temp_bert_token_idx_prompt = bert_token_idx_prompt[bert_clause_idx_prompt_CLS[i]:]
                temp_bert_token_idx = bert_token_idx[bert_clause_idx[i]:]
                if len(temp_bert_token_idx_prompt) <= 512 and len(temp_bert_token_idx) <= 512:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[cls_idx:]
                    bert_segments_idx = bert_segments_idx[cls_idx:]
                    bert_clause_idx = [p - cls_idx for p in bert_clause_idx[i:]]

                    cls_idx_prompt = bert_clause_idx_prompt_CLS[i]
                    bert_clause_idx_prompt = [p - cls_idx_prompt for p in bert_clause_idx_prompt[j:]]
                    bert_token_idx_prompt = bert_token_idx_prompt[cls_idx_prompt:]
                    bert_segments_idx_prompt = bert_segments_idx_prompt[cls_idx_prompt:]
                    bert_clause_idx_prompt_CLS = [p - cls_idx_prompt for p in bert_clause_idx_prompt_CLS[i:]]

                    doc_couples = [[emotion - i, cause - i]]
                    y_emo_id = [id-i for id in y_emo_id]

                    y_emotions = y_emotions[i:]
                    y_causes = y_causes[i:]
                    y_pairs = y_pairs[i:]
                    doc_len = doc_len - i
                    break
                i = i + 1
                j = j + len(self.prompt)
        else:
            i = doc_len - 1
            j = (doc_len - 1) * len(self.prompt)
            while True:
                temp_bert_token_idx_prompt = bert_token_idx_prompt[:bert_clause_idx_prompt_CLS[i]]
                temp_bert_token_idx = bert_token_idx[:bert_clause_idx[i]]
                if len(temp_bert_token_idx_prompt) <= 512 and len(temp_bert_token_idx) <= 512:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[:cls_idx]
                    bert_segments_idx = bert_segments_idx[:cls_idx]
                    bert_clause_idx = bert_clause_idx[:i]

                    cls_idx_prompt = bert_clause_idx_prompt_CLS[i]
                    # print("j", j)
                    bert_clause_idx_prompt = bert_clause_idx_prompt[:j]
                    bert_token_idx_prompt = bert_token_idx_prompt[:cls_idx_prompt]
                    bert_segments_idx_prompt = bert_segments_idx_prompt[:cls_idx_prompt]
                    bert_clause_idx_prompt_CLS = bert_clause_idx_prompt_CLS[:i]

                    y_emotions = y_emotions[:i]
                    y_causes = y_causes[:i]
                    y_pairs = y_pairs[:i]
                    doc_len = i
                    break
                i = i - 1
                j = j - len(self.prompt)
        return bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens, \
               bert_token_idx_prompt, bert_clause_idx_prompt, bert_clause_idx_prompt_CLS, bert_segments_idx_prompt, bert_token_lens_prompt, \
               doc_couples, y_emo_id, y_emotions, y_causes, y_pairs, doc_len





# dataset = MyDataset(fold_id=1, data_type='test')


def bert_batch_preprocessing(batch):
    doc_couples_b, y_emo_id_b, y_emotions_b, y_causes_b, y_pairs_b, doc_len_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_clause_b, bert_token_lens_b, \
    bert_token_prompt_b, bert_segment_prompt_b, bert_clause_prompt_b, bert_token_lens_prompt_b = zip(*batch)

    y_mask_b, y_emotions_b, y_causes_b, y_pairs_b = pad_docs(doc_len_b, y_emotions_b, y_causes_b, y_pairs_b)
    # print("y_mask_b: ", y_mask_b)
    # adj_b = pad_matrices(doc_len_b)
    bert_token_b = pad_sequence(bert_token_b, batch_first=True, padding_value=0)    # 序列填充
    bert_segment_b = pad_sequence(bert_segment_b, batch_first=True, padding_value=0)
    bert_clause_b = pad_sequence(bert_clause_b, batch_first=True, padding_value=0)

    bert_token_prompt_b = pad_sequence(bert_token_prompt_b, batch_first=True, padding_value=0)  # 序列填充
    bert_segment_prompt_b = pad_sequence(bert_segment_prompt_b, batch_first=True, padding_value=0)
    bert_clause_prompt_b = pad_sequence(bert_clause_prompt_b, batch_first=True, padding_value=0)

    bsz, max_len = bert_token_b.size()
    bert_masks_b = np.zeros([bsz, max_len], dtype=np.float)
    for index, seq_len in enumerate(bert_token_lens_b):
        bert_masks_b[index][:seq_len] = 1
    # print("bert_masks_b", bert_masks_b)
    # print(bert_token_prompt_b)

    bsz, max_len = bert_token_prompt_b.size()
    bert_masks_prompt_b = np.zeros([bsz, max_len], dtype=np.float)
    for index, seq_len in enumerate(bert_token_lens_prompt_b):
        bert_masks_prompt_b[index][:seq_len] = 1
    # print("bert_masks_prompt_b", bert_masks_prompt_b)
    # print(bert_token_prompt_b)

    bert_masks_b = torch.FloatTensor(bert_masks_b)
    bert_masks_prompt_b = torch.FloatTensor(bert_masks_prompt_b)

    assert bert_segment_b.shape == bert_token_b.shape
    assert bert_segment_b.shape == bert_masks_b.shape
    assert bert_segment_prompt_b.shape == bert_token_prompt_b.shape
    assert bert_segment_prompt_b.shape == bert_masks_prompt_b.shape


    return np.array(doc_len_b), \
           y_emo_id_b, np.array(y_emotions_b), np.array(y_causes_b), np.array(y_pairs_b), np.array(y_mask_b), doc_couples_b, doc_id_b, \
           bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, \
           bert_token_prompt_b, bert_segment_prompt_b, bert_masks_prompt_b, bert_clause_prompt_b


# 将y_emotions_b, y_causes_b的剩余部分用-1填充，同时构造mask
def pad_docs(doc_len_b, y_emotions_b, y_causes_b, y_pairs_b):
    max_doc_len = max(doc_len_b)

    y_mask_b, y_emotions_b_, y_causes_b_, y_pairs_b_= [], [], [], []
    for y_emotions, y_causes, y_pairs in zip(y_emotions_b, y_causes_b, y_pairs_b):
        y_emotions_ = pad_list(y_emotions, max_doc_len, -1)
        y_causes_ = pad_list(y_causes, max_doc_len, -1)
        y_pairs_ = pad_list(y_pairs, max_doc_len, -1)
        y_mask = list(map(lambda x: 0 if x == -1 else 1, y_emotions_))
        # print("y_mask", y_mask)

        y_mask_b.append(y_mask)
        y_emotions_b_.append(y_emotions_)
        y_causes_b_.append(y_causes_)
        y_pairs_b_.append(y_pairs_)


    return y_mask_b, y_emotions_b_, y_causes_b_, y_pairs_b_

# 将序列的剩余部分用对应字符填充
def pad_list(element_list, max_len, pad_mark):
    element_list_pad = element_list[:]
    pad_mark_list = [pad_mark] * (max_len - len(element_list))
    element_list_pad.extend(pad_mark_list)
    return element_list_pad