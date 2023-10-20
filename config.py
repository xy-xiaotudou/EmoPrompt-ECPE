import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 129

DATA_DIR = './dataset/eca_ch'
TRAIN_FILE = 'fold%s_train.txt'
# VALID_FILE = 'fold%s_valid.txt'
TEST_FILE = 'fold%s_test.txt'
SAMPLER_FILE = 'sampler_data.txt'
# Storing all clauses containing sentimental word, based on the ANTUSD lexicon 'opinion_word_simplified.csv'. see https://academiasinicanlplab.github.io
# SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses.pkl'


class Config(object):
    def __init__(self):
        self.split = 'split10'

        self.bert_cache_path = './bert_base_ch'
        self.fig_path = './test_img'
        self.feat_dim = 768

        self.gnn_dims = '192'
        self.att_heads = '4'
        self.K = 12
        self.pos_emb_dim = 50
        self.pairwise_loss = False

        self.epochs = 20
        self.lr = 1e-5
        self.batch_size = 2
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.l2 = 1e-5
        self.l2_bert = 0.01
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8

        self.classifier_dropout = 0.3
        self.hidden_dropout_prob = 0.3
        self.emotion_classes = 2
        self.cause_classes = 2

        # 原因句提取prompt
        self.total_prompt = {
            "emotion_prompt": {"prompt": "蕴含了[MASK]情感"},    # 没有、一般
            "cause_prompt": {"prompt": "我感觉事情[MASK]", "labels": ["不变", "改变"]},
            "relation_prompt": {"prompt": "[MASK]句子相关"}     # labels：无、一、二、三、四。。。。。
        }

        # 原因词数量
        self.emotion_labels_num = {
            'none': 1,
            'happiness': 48,
            'sadness': 61,
            'fear': 22,
            'anger': 17,
            'disgust': 36,
            'surprise': 9,
        }
