import pickle

def loadList(path):
    pkl_file = open(path, 'rb')
    segContent = pickle.load(pkl_file)
    pkl_file.close()
    return segContent

def saveList(paraList, path):
    output = open(path, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(paraList, output)
    output.close()


def load_data(data_path):
    all_data = []

    inputFile = open(data_path, 'r')
    while True:
        doc_lines = []
        doc_i_dict = dict()
        line = inputFile.readline()
        if line == '':
            break
        line = line.strip().split()
        doc_id = line[0]
        doc_i_dict['doc_id'] = doc_id
        doc_len = int(line[1])
        doc_i_dict['doc_len'] = doc_len

        pairs = eval('[' + inputFile.readline().strip() + ']')
        e_list, c_list = [], []
        for ecp in pairs:
            if ecp[0] not in e_list:
                e_list.append(ecp[0])
            if ecp[1] not in c_list:
                c_list.append(ecp[1])
        doc_i_dict['pairs'] = pairs
        doc_lines.append(doc_i_dict)

        for i in range(doc_len):
            dic = {}
            dic['id'] = i + 1
            if i + 1 in e_list:
                dic['emotion'] = 1
            else:
                dic['emotion'] = 0
            if i + 1 in c_list:
                dic['cause'] = 1
            else:
                dic['cause'] = 0
            contents = (inputFile.readline().strip().split(',')[-1]).replace(' ', '')
            dic['contents'] = contents
            doc_lines.append(dic)
        all_data.append(doc_lines)
    return all_data


txt_data_path = './1.txt' # './all_data_pair.txt'
all_data = load_data(txt_data_path)

save_path = './1.pkl'#'./eca_ch_data_all.pkl'
saveList(all_data, save_path)

# all_text = loadList('./eca_ch_data_all.pkl')
# print(all_text, len(all_text))
#
# dev_data_id = loadList('./split_data_fold/eca_ch_dev_id.pkl')
# # print('dev_data_id:', dev_data_id)
'''
[{'docID': 0}, {'name': 'happiness', 'value': '3'}, 
 [{'key-words-begin': '0', 'keywords-length': '2', 'keyword': '激动', 'clauseID': 3, 'keyloc': 2}], 
 [{'id': '1', 'type': 'v', 'begin': '43', 'length': '11', 'index': 1, 'cause_content': '接受并采纳过的我的建议', 'clauseID': 5}], 
 [{'id': '1', 'cause': 'N', 'keywords': 'N', 'clauseID': 1, 'content': '河北省邢台钢铁有限公司的普通工人白金跃，', 'cause_content': '', 'dis': -2}, 
  {'id': '2', 'cause': 'N', 'keywords': 'N', 'clauseID': 2, 'content': '拿着历年来国家各部委反馈给他的感谢信，', 'cause_content': '', 'dis': -1}, 
  {'id': '3', 'cause': 'N', 'keywords': 'Y', 'clauseID': 3, 'content': '激动地对中新网记者说。', 'cause_content': '', 'dis': 0}, 
  {'id': '4', 'cause': 'N', 'keywords': 'N', 'clauseID': 4, 'content': '“27年来，', 'cause_content': '', 'dis': 1}, 
  {'id': '5', 'cause': 'Y', 'keywords': 'N', 'clauseID': 5, 'content': '国家公安部、国家工商总局、国家科学技术委员会科技部、卫生部、国家发展改革委员会等部委均接受并采纳过的我的建议', 'cause_content': '接受并采纳过的我的建议', 'dis': 2}]
 ]
'''