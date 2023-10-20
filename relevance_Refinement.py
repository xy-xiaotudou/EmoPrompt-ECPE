CUDA_VISIBLE_DEVICES=0

from tqdm import tqdm
import torch
import argparse
import numpy as np

from data_loader import *
from config import *
from network.mtc_model import *
from refinement.filter_method import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
configs = Config()

# 加载模型，词表
model = AutoModelForMaskedLM.from_pretrained(configs.bert_cache_path).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(configs.bert_cache_path)
sampler_loader = create_sampler_data_loader(configs, data_type='sampler')

verbalizerFile = open('refinement/prompt_verbalizer.txt', 'r')
emotion_labels = {}
emotion_labels['none'] = []
emotion_labels['happiness'] = []
emotion_labels['sadness'] = []
emotion_labels['fear'] = []
emotion_labels['anger'] = []
emotion_labels['disgust'] = []
emotion_labels['surprise'] = []
for line in verbalizerFile.readlines():
    line = line.strip().split(" ")
    if line[0] == 'none':
        emotion_labels['none'].extend(line[1:])
        configs.emotion_labels_num['none'] = len(emotion_labels['none'])
    if line[0] == 'happiness':
        emotion_labels['happiness'].extend(line[1:])
        configs.emotion_labels_num['happiness'] = len(emotion_labels['happiness'])
    if line[0] == 'sadness':
        emotion_labels['sadness'].extend(line[1:])
        configs.emotion_labels_num['sadness'] = len(emotion_labels['sadness'])
    if line[0] == 'fear':
        emotion_labels['fear'].extend(line[1:])
        configs.emotion_labels_num['fear'] = len(emotion_labels['fear'])
    if line[0] == 'anger':
        emotion_labels['anger'].extend(line[1:])
        configs.emotion_labels_num['anger'] = len(emotion_labels['anger'])
    if line[0] == 'disgust':
        emotion_labels['disgust'].extend(line[1:])
        configs.emotion_labels_num['disgust'] = len(emotion_labels['disgust'])
    if line[0] == 'surprise':
        emotion_labels['surprise'].extend(line[1:])
        configs.emotion_labels_num['surprise'] = len(emotion_labels['surprise'])

# 更新configs中的prompt的label
emotion_prompt_labels = []
for value in emotion_labels.values():
    emotion_prompt_labels.extend(value)
configs.total_prompt['emotion_prompt']['labels'] = emotion_prompt_labels
# print('total_prompt', configs.total_prompt)

# 获得情感句[mask]处，所有词的词编码
all_logits = []
model.eval()
for batch in tqdm(sampler_loader,desc='ContextCali'):
    doc_len_b, y_emo_id_b, y_emotions_b, y_causes_b, y_pairs_b, y_mask_b, doc_couples_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, \
    bert_token_prompt_b, bert_segment_prompt_b, bert_masks_prompt_b, bert_clause_prompt_b = batch

    # print('y_emo_id_b', y_emo_id_b)
    # print('y_emotions_b', y_emotions_b)



    bert_output_prompt = model(input_ids=bert_token_prompt_b.to(DEVICE),
                                      token_type_ids=bert_segment_prompt_b.to(DEVICE),
                                      attention_mask=bert_masks_prompt_b.to(DEVICE))

    # 提取句子id
    emotion_output = []
    for j in range(bert_clause_prompt_b.shape[0]):
        for i in range(len(y_emo_id_b[j])):
            emotion_output.append(bert_output_prompt.logits.detach()[j, bert_clause_prompt_b[j][y_emo_id_b[j][i]], :])
    emotion_output = torch.stack(emotion_output, 0)
    # print('emotion_output.shape', emotion_output.shape)
    # print('emotion_output', emotion_output)
    all_logits.append(emotion_output)
# print('all_logits', all_logits)
all_logits = torch.cat(all_logits, dim=0)
print('all_logits', all_logits)
print('all_logits.shape', all_logits.shape)

print('emotion_labels', emotion_labels)
label_words = []
for words in emotion_labels.values():
    label_words.append(words)
label_words.pop(0)
print('label_words', label_words)

# 将label_words转化为对应的索引
label_words_index = []
for sub_label_words in label_words:
    label_words_index.append([tokenizer.convert_tokens_to_ids(w) for w in sub_label_words])
print('label_words_index', label_words_index)

# 构建标签词的掩码
label_words_mask = []
lengthMax = len(max(label_words_index, key=len))
for i in range(len(label_words_index)):
    temp_label_words_mask = [1] * len(label_words_index[i])     # 有词的部分为1
    temp_label_words_mask.extend([0] * (lengthMax - len(label_words_index[i])))     # 没有词的部分为0
    label_words_mask.append(temp_label_words_mask)

record = tfidf_filter(all_logits, label_words, label_words_index, torch.tensor(label_words_mask))
print('record', record)

