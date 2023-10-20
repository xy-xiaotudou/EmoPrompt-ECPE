import torch
from torch import nn
from config import DEVICE
import transformers
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# from FocalLoss import *
import numpy

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list):
        super().__init__()
        cls_num_list = torch.tensor(cls_num_list)
        cls_prior = cls_num_list / sum(cls_num_list)
        self.log_prior = torch.log(cls_prior).unsqueeze(0)
        # self.min_prob = 1e-9
        # print(f'Use BalancedSoftmaxLoss, class_prior: {cls_prior}')

    def forward(self, logits, labels):
        adjusted_logits = logits + self.log_prior
        label_loss = F.cross_entropy(adjusted_logits, labels)

        return label_loss

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=7, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim = 1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)
        # print('probs', probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class Multi_Task_ClassifierModel(nn.Module):
    def __init__(self, configs):
        super(Multi_Task_ClassifierModel, self).__init__()
        # self.bert_model = AutoModel.from_pretrained(configs.bert_cache_path)
        self.bert_model_prompt = AutoModelForMaskedLM.from_pretrained(configs.bert_cache_path)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(configs.bert_cache_path)
        classifier_dropout = (
            configs.classifier_dropout
            if configs.classifier_dropout is not None
            else configs.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # 原因词数量
        self.emotion_labels_num = {}
        self.emotion_labels_num['none'] = configs.emotion_labels_num['none']
        self.emotion_labels_num['happiness'] = configs.emotion_labels_num['happiness']
        self.emotion_labels_num['sadness'] = configs.emotion_labels_num['sadness']
        self.emotion_labels_num['fear'] = configs.emotion_labels_num['fear']
        self.emotion_labels_num['anger'] = configs.emotion_labels_num['anger']
        self.emotion_labels_num['disgust'] = configs.emotion_labels_num['disgust']
        self.emotion_labels_num['surprise'] = configs.emotion_labels_num['surprise']
        # print(self.emotion_labels_num)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.bert_model_prompt.config.hidden_size, self.bert_model_prompt.config.hidden_size),
            torch.nn.ReLU(),
            # nn.Dropout(p=args.dropout_prob),
            torch.nn.Linear(self.bert_model_prompt.config.hidden_size, self.bert_model_prompt.config.hidden_size),
        )

        self.extra_token_embeddings = nn.Embedding(5, self.bert_model_prompt.config.hidden_size)

    def forward(self, bert_token_idx, bert_segments_idx, bert_masks_idx, bert_clause_idx, bert_token_idx_prompt, bert_segments_idx_prompt, bert_masks_idx_prompt, bert_clause_idx_prompt, prompt_word_ids):
        # print(bert_token_idx_prompt.shape[0], bert_token_idx_prompt.shape[1])
        # print(bert_segments_idx_prompt.shape[0], bert_segments_idx_prompt.shape[1])

        # 计算句子数
        clause_id = [num_to_chinese4(i+1) for i in range(bert_clause_idx.shape[1])]
        # 1 2 3 ... 10
        clause_id_relation = ['没有']
        clause_id_relation.extend(clause_id)

        clause_id_relation = [self.bert_tokenizer.convert_tokens_to_ids(w) for w in clause_id_relation]

        # prompt的模型输出
        bert_output_prompt = self.bert_model_prompt(input_ids=bert_token_idx_prompt.to(DEVICE),
                                      token_type_ids=bert_segments_idx_prompt.to(DEVICE),
                                      attention_mask=bert_masks_idx_prompt.to(DEVICE))


        # 提取句子id
        emotion_output = []
        cause_output = []
        relation_output = []
        for j in range(bert_clause_idx_prompt.shape[0]):
            emotion_num = 0
            cause_num = 1
            relation_num = 2

            emotion_row = []
            cause_row = []
            relation_row = []

            while(emotion_num<len(bert_clause_idx_prompt[j])):
                # print(len(bert_clause_idx_prompt[j]))
                # print(bert_clause_idx_prompt[j][relation_num])
                emotion_row.append(bert_output_prompt.logits[j, bert_clause_idx_prompt[j][emotion_num], prompt_word_ids[0]])
                cause_row.append(bert_output_prompt.logits[j, bert_clause_idx_prompt[j][cause_num], prompt_word_ids[1]])
                relation_row.append(bert_output_prompt.logits[j, bert_clause_idx_prompt[j][relation_num], clause_id_relation])

                emotion_num += len(prompt_word_ids)
                cause_num += len(prompt_word_ids)
                relation_num += len(prompt_word_ids)

            emotion_output.append( torch.stack(emotion_row, 0))
            cause_output.append(torch.stack(cause_row, 0))
            relation_output.append(torch.stack(relation_row, 0))


        emotion_output = torch.stack(emotion_output, 0)
        emotion_output_ = []
        index = 0
        # print(emotion_output.shape[1])
        while index<emotion_output.shape[2]:
            for labels_num in self.emotion_labels_num.values():
                # print('emotion_output[:, :, index:index+labels_num]', emotion_output[:, :, index:index+labels_num])
                sum = torch.div(torch.sum(emotion_output[:, :, index:index+labels_num], dim=2), labels_num)
                # print('sum', sum)
                emotion_output_.append(sum)
                index += labels_num
        emotion_output = torch.stack(emotion_output_, 2)
        cause_output = torch.stack(cause_output, 0)
        relation_output = torch.stack(relation_output, 0)

        return emotion_output, cause_output, relation_output


    def loss_pre(self, pred_e, pred_c, pred_relation, y_emotions, y_causes, y_pairs, y_mask, emotion_num_list):
        # y_mask = torch.ByteTensor(y_mask).to(DEVICE)
        y_emotions = torch.FloatTensor(y_emotions).to(DEVICE)
        y_causes = torch.FloatTensor(y_causes).to(DEVICE)
        y_pairs = torch.FloatTensor(y_pairs).to(DEVICE)

        criterion = CrossEntropyLoss()
        criterion_e = FocalLoss()
        # 计算emotion的loss
        pred_e = pred_e.view(-1, pred_e.shape[2])
        y_emotions = y_emotions.view(-1)


        del_index = [i for i in range(y_emotions.shape[0]) if y_emotions[i] == -1]
        for i in range(len(del_index)):
            pred_e = pred_e[torch.arange(pred_e.size(0)) != del_index[i] - i]
            y_emotions = y_emotions[torch.arange(y_emotions.size(0)) != del_index[i] - i]

        new_pred_e = pred_e
        new_y_emotions = y_emotions.long()

        # print("new_y_emotions", new_y_emotions)
        # print("new_pred_e", new_pred_e)
        loss_e = criterion_e(new_pred_e, new_y_emotions)

        # 计算cause的loss
        pred_c = pred_c.view(-1, 2)
        y_causes = y_causes.view(-1)
        del_index = [i for i in range(y_causes.shape[0]) if y_causes[i] == -1]
        for i in range(len(del_index)):
            pred_c = pred_c[torch.arange(pred_c.size(0)) != del_index[i] - i]
            y_causes = y_causes[torch.arange(y_causes.size(0)) != del_index[i] - i]

        new_pred_c = pred_c
        new_y_causes = y_causes.long()
        loss_c = criterion(new_pred_c, new_y_causes)

        # 计算relation的loss
        pred_relation = pred_relation.view(-1, pred_relation.shape[2])
        y_pairs = y_pairs.view(-1)
        # print("y_pairs", y_pairs)
        # print("pred_relation", pred_relation)
        del_index = [i for i in range(y_pairs.shape[0]) if y_pairs[i] == -1]
        for i in range(len(del_index)):
            pred_relation = pred_relation[torch.arange(pred_relation.size(0)) != del_index[i] - i]
            y_pairs = y_pairs[torch.arange(y_pairs.size(0)) != del_index[i] - i]

        new_pred_relation = pred_relation
        new_y_relation = y_pairs.long()
        loss_relation = criterion(new_pred_relation, new_y_relation)




        return loss_e, loss_c, loss_relation

