# EmoPrompt-ECPE
文件：
config.py：配置文件（模型参数等）
data_loader.py：数据的读取和加载文件
main.py：程序运行文件（直接运行python main.py）
relevance_Refinement.py：词库筛选文件（通过运行该文件进行词库的筛选）

文件夹：
bert_base_ch：存放预训练模型
dataset：存放数据集
network：存放模型
	mtc_model.py：总的模型架构文件
	wobert_tokenizer.py：wobert的tokenizer
refinement：存放词库
	filter_method.py：筛选词库的算法（该文件会通过relevance_Refinement.py文件调用，不用主动运行）
