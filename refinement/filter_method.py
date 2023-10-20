import numpy
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 



def tfidf_filter(cc_logits, class_labels, label_words_index, label_words_mask):
    myrecord = ""
    class_num = len(class_labels)
    norm_ord = 10/(class_num-2+1e-2) +1 
    print("norm_ord", norm_ord)
    context_size = cc_logits.shape[0]   # [200, 50265]
    # 得到词表中每个词在200个句子中的分布
    tobeproject = cc_logits.transpose(0,1)     # transpose(Tensor,dim0,dim1)是矩阵转置操作，unsqueeze()是矩阵升维操作
    print('tobeproject', tobeproject.size())

    # 提取label_words对应的分数（对齐 构建向量）
    ret = []
    lengthMax = len(max(label_words_index, key=len))
    for i in range(len(label_words_index)):
        temp_label_words_index = label_words_index[i]
        temp_label_words_index.extend([0] * (lengthMax - len(label_words_index[i])))
        ret.append(tobeproject[temp_label_words_index, :])
    ret = torch.stack(ret, dim=0)    # 在给定维度进行拼接的操作
    print('ret', ret.size())    # [1, 2, 266, 200]，2是因为有两列标签词汇（分别是good和bad）


    label_words_cc_logits = ret
    print('label_words_cc_logits', label_words_cc_logits.size())

    label_words_cc_logits = label_words_cc_logits - label_words_cc_logits.mean(dim=-1,keepdims=True)#, dim=-1)

    # 将第一个标签词所对应的向量进行提取
    first_label_logits = label_words_cc_logits[:,0,:]
    orgshape = label_words_cc_logits.shape      # [2, 266, 200]
    label_words_cc_logits = label_words_cc_logits.reshape(-1,context_size)
    print('label_words_cc_logits', label_words_cc_logits.size())

    sim_mat = cosine_similarity(label_words_cc_logits.cpu().numpy(),first_label_logits.cpu().numpy() ).reshape(*orgshape[:-1],first_label_logits.shape[0])
    sim_mat = sim_mat - 10000.0 * (1-label_words_mask.unsqueeze(-1).cpu().numpy())
    print('myverbalizer.label_words_mask', label_words_mask.shape)

    new_label_words = []
    max_lbw_num_pclass = label_words_mask.shape[-1]
    outputers = []
    for class_id in range(len(class_labels)):
        tfidf_scores = []
        tf_scores = []
        idf_scores = []
        num_words_in_class = len(class_labels[class_id])
        for in_class_id in range(max_lbw_num_pclass):
            if label_words_mask[class_id, in_class_id] > 0:
                word_sim_scores = sim_mat[class_id, in_class_id]
                tf_score = word_sim_scores[class_id]
                idf_score_source = np.concatenate([word_sim_scores[:class_id], word_sim_scores[class_id+1:]])
                idf_score = 1/ (np.linalg.norm(idf_score_source, ord=norm_ord)/np.power((class_num-1), 1/norm_ord))
                tfidf_score = tf_score * idf_score #+1e-15)
                if tf_score<0:
                    tfidf_score = -100
                tfidf_scores.append(tfidf_score)
                tf_scores.append(tf_score)
                idf_scores.append(idf_score)
    
        outputer = list(zip(class_labels[class_id],
                                            tfidf_scores,
                                            tf_scores,
                                            idf_scores))
        
        outputer = sorted(outputer, key=lambda x:-x[1])
        outputers.append(outputer)
    print('outputers[0]: {}, outputers[1]: {}'.format(len(outputers), len(outputers[0])))
    print('outputers[0][0]', outputers[0][0])

    cut_optimality = []
    max_outputer_len = max([len(outputers[class_id]) for class_id in range(len(outputers))])
    for cut_potent in range(max_outputer_len):
        cut_rate = cut_potent/max_outputer_len
        loss = 0
        for class_id in range(len(class_labels)):
            cut_potent_this_class = int(cut_rate*len(outputers[class_id]))
            if len(outputers[class_id]) <= cut_potent_this_class:
                boundary_score = outputers[class_id][-1][1]
            else:
                boundary_score = outputers[class_id][cut_potent_this_class][1]
            loss += (boundary_score-1)**2
        cut_optimality.append([cut_rate, loss])
    print('cut_optimality', cut_optimality)
    optimal_cut_rate = sorted(cut_optimality, key=lambda x:x[1])[0][0]
    print("optimal_cut rate is {}".format(optimal_cut_rate))
    for class_id in range(len(class_labels)):
        cut = int(len(outputers[class_id])*optimal_cut_rate)
        if cut==0:
            cut=1
        # cut = optimal_cut
        new_l = [x[0] for x in outputers[class_id][:cut]]
        removed_words = [x[0] for x in outputers[class_id][cut:]]
        myrecord += f"Class {class_id} {new_l}\n"
        myrecord +=f"Class {class_id} rm: {removed_words}\n"
        new_label_words.append(new_l)
    class_labels = new_label_words
    # myverbalizer = myverbalizer.cuda()
    noww_label_words_num = [len(class_labels[i]) for i in range(len(class_labels))]
    myrecord += f"Phase 3 {noww_label_words_num}\n"

    print('class_labels', class_labels)

    # 将提取后的label_words写入文件
    emotion_labels = {}
    emotion_labels['happiness'] = class_labels[0]
    emotion_labels['sadness'] = class_labels[1]
    emotion_labels['fear'] = class_labels[2]
    emotion_labels['anger'] = class_labels[3]
    emotion_labels['disgust'] = class_labels[4]
    emotion_labels['surprise'] = class_labels[5]
    print(emotion_labels)
    verbalizerFile = open('prompt_verbalizer_Refinement.txt', 'w')
    verbalizerFile.write('none' + ' 没有' + '\n')
    for key, value in emotion_labels.items():
        verbalizerFile.write(key + ' ')
        for emotion_word in value:
            verbalizerFile.write(emotion_word + ' ')
        verbalizerFile.write('\n')

    return myrecord


