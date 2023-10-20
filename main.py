import torch
import json
import transformers
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import *
from config import *
from network.mtc_model import *
# from network.FocalLoss import *

TORCH_SEED = 2022
DATA_PATH = './dataset/eca_ch'
results_path = './results.txt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file = open(results_path, 'w')

def main(configs):
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    # ECA_CH_processor = ECA_CH_Processor()
    # label_list = ECA_CH_processor.get_labels()
    # print(label_list)
    metric_folds = {
        'emo': [],
        'cau': [],
        'relation': []
    }
    # file = open(results_path, 'w')

    # 读取标签词
    verbalizerFile = open('refinement/prompt_verbalizer_Refinement_best_utf8.txt', 'r')
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
    # print(emotion_labels)
    # for value in emotion_labels.values():
    #     print(len(value))
    # print(configs.emotion_labels_num)
    # print('total_prompt(pre)', configs.total_prompt)
    emotion_prompt_labels = []
    for value in emotion_labels.values():
        emotion_prompt_labels.extend(value)
    configs.total_prompt['emotion_prompt']['labels'] = emotion_prompt_labels
    # print('total_prompt', configs.total_prompt)

    for fold_id in range(1, 11):
        print('-'*15, 'fold{}'.format(fold_id), '-'*15)
        file.write('-'*15 + 'fold{}'.format(fold_id) + '-'*15 + '\n')

        # 导入数据
        train_loader = create_train_data_loader(configs, fold_id)
        test_loader = create_test_data_loader(configs, fold_id, 'test')
        # print("train_loader: ", len(train_loader))
        # print("test_loader: ", len(test_loader))
        # 取出label word转成id
        bert_tokenizer = AutoTokenizer.from_pretrained(configs.bert_cache_path)
        prompt_word_ids = []
        for sub_prompt in configs.total_prompt:
            if "labels" in configs.total_prompt[sub_prompt]:
                prompt_word_ids.append([bert_tokenizer.convert_tokens_to_ids(w) for w in configs.total_prompt[sub_prompt]['labels']])
            else:
                prompt_word_ids.append([])
        print(prompt_word_ids)

        # 计算该fold中每一类的样本数量
        fold_path = DATA_PATH + '/' + join('fold%s_train.txt' % fold_id)
        emotion_num_list = get_emotion_num_list(configs, fold_path)
        # print(emotion_num_list)

        model = Multi_Task_ClassifierModel(configs).to(DEVICE)
        print(DEVICE)

        # for p in model.named_parameters():
        #     print('model.named_parameters()', p[0])

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': configs.l2_bert, 'eps': configs.adam_epsilon},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'eps': configs.adam_epsilon}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=configs.lr)

        num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
        warmup_steps = int(num_steps_all * configs.warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_steps_all)
        model.zero_grad()
        # max_ec, max_e, max_c = (-1, -1, -1), None, None
        max_metric_e, max_metric_c, max_metric_relation = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}, \
                             {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}, \
                             {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}
        
        metric_folds['emo'].append(max_metric_e)
        metric_folds['cau'].append(max_metric_c)
        metric_folds['relation'].append(max_metric_relation)
        
        early_stop_flag = None

        for epoch in range(1, configs.epochs + 1):
            step = 1
            for train_step, batch in enumerate(train_loader, 1):
                model.train()
                doc_len_b, y_emo_id_b, y_emotions_b, y_causes_b, y_pairs_b, y_mask_b, doc_couples_b, doc_id_b, \
                bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, \
                bert_token_prompt_b, bert_segment_prompt_b, bert_masks_prompt_b, bert_clause_prompt_b = batch

                pred_e, pred_c, pred_relation = model(bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, bert_token_prompt_b, bert_segment_prompt_b, bert_masks_prompt_b, bert_clause_prompt_b, prompt_word_ids)


                loss_e, loss_c, loss_relation = model.loss_pre(pred_e, pred_c, pred_relation, y_emotions_b, y_causes_b, y_pairs_b, y_mask_b, emotion_num_list)
                # print(pred_e, pred_c)
                # print(y_causes_b)
                loss = loss_e + loss_c + loss_relation
                loss = loss / configs.gradient_accumulation_steps
                loss.requires_grad_(True)

                loss.backward()
                if train_step % configs.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if step % 10 == 0:
                    print('Fold {}, Epoch {}, step {}, loss {}'.format(fold_id, epoch, step, loss))

                    file.write('Fold {}, Epoch {}, step {}, loss {}'.format(fold_id, epoch, step, loss) + '\n')
                    # file.write('pred_e:' + json.dumps(pred_e.tolist()) + '\n')
                    # print(y_ids_b)
                    emotions_metrics = compute_metrics(pred_e.cpu(), y_emotions_b)
                    print('emotions_metrics:', emotions_metrics)
                    file.write('emotions_metrics:' + json.dumps(emotions_metrics) + '\n')
                    cause_metrics = compute_metrics(pred_c.cpu(), y_causes_b)
                    print('cause_metrics:', cause_metrics)
                    file.write('cause_metrics:' + json.dumps(cause_metrics) + '\n')
                    relation_metrics = compute_metrics(pred_relation.cpu(), y_pairs_b)
                    print('relation_metrics:', relation_metrics)
                    file.write('relation_metrics:' + json.dumps(relation_metrics) + '\n')
                    # 画混淆矩阵
                    # confusionMatrix(pred_e.cpu(), y_emotions_b)
                step += 1
            file.write('-'*15 + 'test' + '-'*15 + '\n')
            ## test
            with torch.no_grad():
                model.eval()

                # 计算该fold中每一类的样本数量
                fold_path = DATA_PATH + '/' + join('fold%s_test.txt' % fold_id)
                emotion_num_list = get_emotion_num_list(configs, fold_path)
                # print(emotion_num_list)

                emotions_metrics, cause_metrics, relation_metrics,  pred_list, true_lsit = inference_one_epoch(configs, test_loader, model, emotion_num_list)
                print('emotions_metrics, cause_metrics, relation_metrics:', emotions_metrics, cause_metrics, relation_metrics)
                file.write('emotions_metrics:' + json.dumps(emotions_metrics) + '\n')
                file.write('cause_metrics:' + json.dumps(cause_metrics) + '\n')
                file.write('relation_metrics:' + json.dumps(relation_metrics) + '\n')

                # print('pred_list', pred_list)
                # print('true_lsit', true_lsit)

                # 绘制热力图
                fig_path = configs.fig_path + '/' + join('Fold%s' % fold_id) + join('_epoch%s.png' % epoch)
                print('save to: ', fig_path)
                sns.set()
                f, ax = plt.subplots()
                C2 = confusionMatrix(pred_list, true_lsit)
                ax.set_title('confusion matrix')  # 标题
                ax.set_xlabel('predict')  # x 轴
                ax.set_ylabel('true')  # y 轴
                fig = sns.heatmap(C2, annot=True, ax=ax)  # 画热力图

                scatter_fig = fig.get_figure()
                scatter_fig.savefig(fig_path, dpi=400)
                
                
                
                

                if relation_metrics['f1'] > max_metric_relation['f1']:
                    early_stop_flag = 1
                    max_metric_e, max_metric_c, max_metric_relation = emotions_metrics, cause_metrics, relation_metrics
                    
                    metric_folds['emo'].append(emotions_metrics)
                    metric_folds['cau'].append(cause_metrics)
                    metric_folds['relation'].append(relation_metrics)
                    
                    torch.save(model.state_dict(),
                               'best_model' + '_fold' + str(fold_id) + '_epoch' + str(epoch) +
                               '_step' + str(step) + '.bin')
                else:
                    early_stop_flag += 1
    for key, value in metric_folds.items():
        file.write(key + ' ')
        for metric in value:
            file.write(json.dumps(metric) + '\n')



# 画混淆矩阵
def confusionMatrix(predictions, labels):
    preds = predictions
    new_labels, new_preds = [], []
    for i in range(len(labels)):
        if labels[i] != -1:
            new_labels.append(labels[i])
            new_preds.append(preds[i])

    C2 = confusion_matrix(new_labels, new_preds, labels=[0, 1, 2, 3, 4, 5, 6])
    # 打印 C2
    print(C2)

    return C2




def compute_metrics(predictions, labels):
    preds = predictions.argmax(-1)
    # print('preds', preds)
    p_list, r_list, f1_list, acc_list = [], [], [], []
    for i in range(labels.shape[0]):
        new_labels, new_preds = [], []
        for j in range(labels.shape[1]):
            if labels[i][j] != -1:
                new_labels.append(labels[i][j])
                new_preds.append(preds[i][j])

        p_batch = precision_score(new_labels, new_preds, average='weighted', zero_division=0)
        r_batch = recall_score(new_labels, new_preds, average='weighted', zero_division=0)
        f1_batch = f1_score(new_labels, new_preds, average='weighted')
        acc_batch = accuracy_score(new_labels, new_preds)


        p_list.append(p_batch)
        r_list.append(r_batch)
        f1_list.append(f1_batch)
        acc_list.append(acc_batch)

    return {
        'accuracy': np.mean(acc_list),
        'precision': np.mean(p_list),
        'recall': np.mean(r_list),
        'f1': np.mean(f1_list)
    }


def inference_one_batch(configs, batch, model, emotion_num_list):
    bert_tokenizer = AutoTokenizer.from_pretrained(configs.bert_cache_path)
    prompt_word_ids = []
    for sub_prompt in configs.total_prompt:
        if "labels" in configs.total_prompt[sub_prompt]:
            prompt_word_ids.append(
                [bert_tokenizer.convert_tokens_to_ids(w) for w in configs.total_prompt[sub_prompt]['labels']])
        else:
            prompt_word_ids.append([])
    doc_len_b, y_emo_id_b, y_emotions_b, y_causes_b, y_pairs_b, y_mask_b, doc_couples_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, \
    bert_token_prompt_b, bert_segment_prompt_b, bert_masks_prompt_b, bert_clause_prompt_b = batch
    pred_e, pred_c, pred_relation = model(bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, bert_token_prompt_b, bert_segment_prompt_b, bert_masks_prompt_b, bert_clause_prompt_b, prompt_word_ids)
    # file.write('pred_e_eval:' + json.dumps(pred_e.tolist()) + '\n')
    loss_e, loss_c, loss_relation = model.loss_pre(pred_e, pred_c, pred_relation, y_emotions_b, y_causes_b, y_pairs_b, y_mask_b, emotion_num_list)
    return loss_e, loss_c, loss_relation, pred_e, pred_c, pred_relation, y_emotions_b, y_causes_b, y_pairs_b


def inference_one_epoch(configs, batches, model, emotion_num_list):
    pred_list, true_lsit = [], []
    p_list_emotion, r_list_emotion, f1_list_emotion, acc_list_emotion = [], [], [], []
    p_list_cause, r_list_cause, f1_list_cause, acc_list_cause = [], [], [], []
    p_list_relation, r_list_relation, f1_list_relation, acc_list_relation = [], [], [], []
    id_metrics_f, emotion_metrics_f, cause_metrics_f, relation_metrics_f = {}, {}, {}, {}
    for batch in batches:
        loss_e, loss_c, loss_relation, pred_emotion, pred_cause, pred_relation, y_emotions_b, y_causes_b, y_pairs_b = inference_one_batch(configs, batch, model, emotion_num_list)
        emotions_metrics = compute_metrics(pred_emotion.cpu(), y_emotions_b)
        p_list_emotion.append(emotions_metrics['precision'])
        r_list_emotion.append(emotions_metrics['recall'])
        f1_list_emotion.append(emotions_metrics['f1'])
        acc_list_emotion.append(emotions_metrics['accuracy'])
        cause_metrics = compute_metrics(pred_cause.cpu(), y_causes_b)
        p_list_cause.append(cause_metrics['precision'])
        r_list_cause.append(cause_metrics['recall'])
        f1_list_cause.append(cause_metrics['f1'])
        acc_list_cause.append(cause_metrics['accuracy'])
        relation_metrics = compute_metrics(pred_relation.cpu(), y_pairs_b)
        p_list_relation.append(relation_metrics['precision'])
        r_list_relation.append(relation_metrics['recall'])
        f1_list_relation.append(relation_metrics['f1'])
        acc_list_relation.append(relation_metrics['accuracy'])
        # 画混淆矩阵
        # print(pred_emotion.argmax(-1).tolist())
        # print(y_emotions_b)
        for i in range(pred_emotion.shape[0]):
            pred_list.extend(pred_emotion.argmax(-1).tolist()[i])
            true_lsit.extend(y_emotions_b[i])


    emotion_metrics_f['accuracy'] = np.mean(acc_list_emotion)
    emotion_metrics_f['precision'] = np.mean(p_list_emotion)
    emotion_metrics_f['recall'] = np.mean(r_list_emotion)
    emotion_metrics_f['f1'] = np.mean(f1_list_emotion)

    cause_metrics_f['accuracy'] = np.mean(acc_list_cause)
    cause_metrics_f['precision'] = np.mean(p_list_cause)
    cause_metrics_f['recall'] = np.mean(r_list_cause)
    cause_metrics_f['f1'] = np.mean(f1_list_cause)

    relation_metrics_f['accuracy'] = np.mean(acc_list_relation)
    relation_metrics_f['precision'] = np.mean(p_list_relation)
    relation_metrics_f['recall'] = np.mean(r_list_relation)
    relation_metrics_f['f1'] = np.mean(f1_list_relation)


    return emotion_metrics_f, cause_metrics_f, relation_metrics_f, pred_list, true_lsit


if __name__ == '__main__':
    configs = Config()
    main(configs)
