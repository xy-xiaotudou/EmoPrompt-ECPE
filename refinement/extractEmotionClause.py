emotion_clause = {}
emotion_clause['happiness'] = []
emotion_clause['sadness'] = []
emotion_clause['fear'] = []
emotion_clause['anger'] = []
emotion_clause['disgust'] = []
emotion_clause['surprise'] = []

# 数据读取
def load_emotion_clause(data_file):
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
                e_list.append(ecp[0])
            if ecp[1] not in c_list:
                c_list.append(ecp[1])
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

            # # 添加prompt的文本
            # dic['clause_prompt'] = contents
            # for sub_prompt in total_prompt:
            #     dic['clause_prompt'] = dic['clause_prompt'] + ',' + total_prompt[sub_prompt]['prompt']
            #     # print("sub_prompt:", sub_prompt)
            # dic['clause_prompt'] = num_to_chinese4(dic['clause_id']) + '、' + dic['clause_prompt'] + '。'
            # print("clause_prompt", dic['clause_prompt'])
            doc_lines.append(dic)
        # print(len(e_list))
        for clause_id in e_list:
            # print(clause_id)
            # print(doc_lines[clause_id-1])
            if doc_lines[clause_id-1]['emotion_category'] == 'happiness':
                emotion_clause['happiness'].append(doc_lines[clause_id-1])
            if doc_lines[clause_id - 1]['emotion_category'] == 'sadness':
                emotion_clause['sadness'].append(doc_lines[clause_id-1])
            if doc_lines[clause_id - 1]['emotion_category'] == 'fear':
                emotion_clause['fear'].append(doc_lines[clause_id-1])
            if doc_lines[clause_id - 1]['emotion_category'] == 'anger':
                emotion_clause['anger'].append(doc_lines[clause_id-1])
            if doc_lines[clause_id - 1]['emotion_category'] == 'disgust':
                emotion_clause['disgust'].append(doc_lines[clause_id-1])
            if doc_lines[clause_id - 1]['emotion_category'] == 'surprise':
                emotion_clause['surprise'].append(doc_lines[clause_id-1])
        doc_i_dict['clauses'] = doc_lines
        all_data.append(doc_i_dict)
        # print('all_data:', all_data)
    return all_data




if __name__ == '__main__':
    dataFile = '../dataset/eca_ch/all_data_pair.txt'
    load_emotion_clause(dataFile)

    # 将所有情感句写进train_emotion_clause.txt文件
    writeFile = open('train_emotion_clause.txt', 'w')
    for key, value in emotion_clause.items():
        writeFile.write('-'*15 + key + '-'*15 + '\n')
        for emotion_dic in value:
            for clause_key, clause_value in emotion_dic.items():
                writeFile.write(clause_key + ': ' + str(clause_value) + '   ')
            writeFile.write('\n')

    # 将情感词进行提取
    vocabFile = open('../bert_base_ch/vocab.txt', 'r', encoding='utf-8')
    bertVocab = []      # bert的词典列表
    for line in vocabFile.readlines():
        line = line.strip().split(" ")
        bertVocab.extend(line)
    # print(bertVocab)

    verbalizerFile = open('prompt_verbalizer.txt', 'w')
    verbalizerFile.write('none' + ' 没有' + '\n')
    for key, value in emotion_clause.items():
        verbalizerFile.write(key + ' ')
        removeDuplication = []
        for emotion_dic in value:
            if (emotion_dic['emotion_token'] in bertVocab) and (emotion_dic['emotion_token'] not in removeDuplication):
                removeDuplication.append(emotion_dic['emotion_token'])
                verbalizerFile.write(emotion_dic['emotion_token'] + ' ')
        verbalizerFile.write('\n')