class HmmTrain:
    def __init__(self):
        self.line_index = -1

    def init(self):  # 初始化字典
        trans_dict = {}  # 存储状态转移概率
        emit_dict = {}  # 发射概率(状态->词语的条件概率)
        count_dict = {}  # 存储所有状态序列 ，用于归一化分母
        start_dict = {}  # 存储状态的初始概率
        state_list = ['B', 'M', 'E', 'S']  # 状态序列

        for state in state_list:
            trans_dict[state] = {}
            for state1 in state_list:
                trans_dict[state][state1] = 0.0
        for state in state_list:
            start_dict[state] = 0.0
            emit_dict[state] = {}
            count_dict[state] = 0

        return trans_dict, emit_dict, start_dict, count_dict

    def save_model(self, word_dict, model_path):
        '''
        保存模型
        '''
        f = open(model_path, 'w')
        f.write(str(word_dict))
        f.close()

    '''词语状态转换(类似打标签)'''

    def get_word_status(self, word):  # 根据词语，输出词语对应的SBME状态
        '''
        S:单字词
        B:词的开头
        M:词的中间
        E:词的末尾
        能 ['S']
        前往 ['B', 'E']
        科威特 ['B', 'M', 'E']
        '''
        word_status = []
        if len(word) == 1:
            word_status.append('S')
        elif len(word) == 2:
            word_status = ['B', 'E']
        else:
            num_m = len(word) - 2
            list_m = ['M'] * num_m
            word_status.append('B')
            word_status.extend(list_m)
            word_status.append('E')
        return word_status

    '''基于人工标注语料库，训练发射概率，初始状态， 转移概率'''

    def train(self, train_filepath, trans_path, emit_path, start_path):
        trans_dict, emit_dict, start_dict, count_dict = self.init()

        for line in open(train_filepath, encoding='utf-8'):
            self.line_index += 1
            line = line.strip()
            if not line:
                continue
            char_list = []
            for i in range(len(line)):
                if line[i] == " ":
                    continue
                char_list.append(line[i])
            word_list = line.split(" ")
            line_status = []
            for word in word_list:
                line_status.extend(self.get_word_status(word))

            if len(line_status) == len(char_list):
                for i in range(len(line_status)):
                    if i == 0:
                        start_dict[line_status[0]] += 1
                        count_dict[line_status[0]] += 1
                    else:
                        trans_dict[line_status[i - 1]][line_status[i]] += 1
                        count_dict[line_status[i]] += 1

                        if char_list[i] not in emit_dict[line_status[i]]:
                            emit_dict[line_status[i]][char_list[i]] = 0.0
                        else:
                            emit_dict[line_status[i]][char_list[i]] += 1
            else:
                continue
        # 做初始状态矩阵的归一化
        for key in start_dict:
            start_dict[key] = start_dict[key] * 1.0 / self.line_index
        # 状态转移矩阵的归一化
        for key in trans_dict:
            for key1 in trans_dict[key]:
                trans_dict[key][key1] = trans_dict[key][key1] / count_dict[key]
        # 发射矩阵的归一化
        for key in emit_dict:
            for word in emit_dict[key]:
                emit_dict[key][word] = emit_dict[key][word] / count_dict[key]

        self.save_model(trans_dict, trans_path)
        self.save_model(emit_dict, emit_path)
        self.save_model(start_dict, start_path)


if __name__ == "__main__":
    train_filepath = '/Users/henry/Documents/application/nlp_assignments/data/rc/train.txt'

    trans_path = '/Users/henry/Documents/application/nlp_assignments/data/rc/model/prob_trans.model'
    emit_path = '/Users/henry/Documents/application/nlp_assignments/data/rc/model/prob_emit.model'
    start_path = '/Users/henry/Documents/application/nlp_assignments/data/rc/model/prob_start.model'
    trainer = HmmTrain()
    trainer.train(train_filepath, trans_path, emit_path, start_path)
