# -*- coding: utf-8 -*-
# @Time    : 2019-12-02 10:36
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : seq2seq.py
# @Description:seq2seq模型
import time
from enum import Enum

from beamSearch import *
from layer import Encoder, Decoder
from util import *


class Seq2seq(tf.keras.Model):
    TYPE = Enum('TYPE', ('TEST', 'TRAIN','EVAL'))
    def __init__(self,type, vocab,embedding_matrix,reverse_vocab=None, train_X=None,train_Y=None,test_X=None,units=1024,modelFile='data/checkpoints/training_checkpoints',BATCH_SIZE = 4,paddingChr = '<PAD>'):
        """
        :param type: 很别扭
        :param train_X: 训练集输入
        :param train_X: 训练集输入
        :param test_X: 测试集输入
        :param modelFile: 词向量模型文件
        :param units: 隐藏层单元数
        :param BATCH_SIZE:
        :param paddingChr:语料中padding的字符
        """
        super(Seq2seq, self).__init__()
        assert type is not None, 'please ch0ose the type of operator : TEST or Train'

        if type == Seq2seq.TYPE.TRAIN:
                assert train_X is not  None,'train_X can not be None '
                assert train_Y is not None, 'train_Y can not be None '
                # 训练集的长度
                self.BUFFER_SIZE = len(train_X)
                # 输入的长度
                self.max_length_inp = train_X.shape[1]
                # 输出的长度
                self.max_length_targ = train_Y.shape[1]

        elif  type == Seq2seq.TYPE.TEST:
                assert test_X is not None, 'test_X can not be None '
                # 测试集的长度
                self.BUFFER_SIZE = len(test_X)
                # 输入的长度
                self.max_length_inp = test_X.shape[1]
                # 输出的长度
                self.max_length_targ = 50
        elif  type == Seq2seq.TYPE.EVAL:
                self.BUFFER_SIZE = 32
                # 输入的长度
                self.max_length_inp = 200
                # 输出的长度
                self.max_length_targ = 50
        print('max_length_inp =  {} max_length_max_targ =  {}  '.format( self.max_length_inp,self.max_length_targ))

        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
        self.embedding_matrix =  embedding_matrix
        # 词向量维度
        self.embedding_dim = self.embedding_matrix.shape[1]
        # 词表大小
        self.vocab_size = len(self.vocab)
        self.pad_index = self.vocab[paddingChr]

        self.BATCH_SIZE  = BATCH_SIZE
        #每一轮的步数，取整除 - 向下取接近除数的整数>>> 9//2  =  4   >>> -9//2  = -5
        self.steps_per_epoch = self.BUFFER_SIZE//BATCH_SIZE
        #隐藏层，单元数
        self.units = units
        #from_logits=True 表示内部就会自动做softmax
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')#稀疏交叉墒

        self.encoder = Encoder(self.vocab_size, self.embedding_dim, self.embedding_matrix, self.units, self.BATCH_SIZE)
        self.decoder = Decoder(self.vocab_size, self.embedding_dim, self.embedding_matrix, self.units,self.BATCH_SIZE)

        self.optimizer = tf.keras.optimizers.Adam()
        #构建训练集
        #dataset = tf.data.Dataset.from_generator()

        if type == Seq2seq.TYPE.TRAIN:
            #数据集不是很大的时候
            self.dataset = tf.data.Dataset.from_tensor_slices((train_X,train_Y)).shuffle(self.BUFFER_SIZE)
            #用于标示是否对于最后一个batch如果数据量达不到batch_size时保留还是抛弃
            self.dataset = self.dataset.batch(self.BATCH_SIZE,drop_remainder=True)
        elif type == Seq2seq.TYPE.TEST:
            self.dataset = tf.data.Dataset.from_tensor_slices(test_X)
            self.dataset = self.dataset.batch(self.BATCH_SIZE)


        self.checkpoint_dir = modelFile+'training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                         encoder=self.encoder,
                                         decoder=self.decoder)

        if os.path.exists(self.checkpoint_dir+ os.sep + 'checkpoint'):
            self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))



    def train(self,debug = False):
        """
        训练函数
        :param debug: 是否是调试模式
        :return:
        """
        minloss = float("inf")
        EPOCHS = 10
        tf.config.experimental_run_functions_eagerly(debug)
        for epoch in range(EPOCHS):
            start = time.time()

            # 初始化隐藏层
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(self.dataset.take(self.steps_per_epoch)):
                #
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 1 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # saving the minloss  model

            curloss = total_loss / self.steps_per_epoch
            if curloss <=  minloss:
                minloss = curloss
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, minloss))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_hidden

    def call_decoder_onestep(self, dec_input, dec_hidden, enc_output):
        """

        :param dec_input:
        :param dec_hidden:
        :param enc_output:
        :return:
        """
        #context_vector, attention_weights = self.attention(dec_hidden, enc_output)
        #pred, dec_hidden = self.decoder(dec_input, None, None,context_vector)
        predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_output)
        return predictions, dec_hidden, attention_weights


    @tf.function
    def train_step(self,inp, targ, enc_hidden):
        """
        一个tf.function定义就像是一个核心TensorFlow操作：可以急切地执行它; 也可以在静态图中使用它; 且它具有梯度。
        :param inp: input
        :param targ: 目标
        :param enc_hidden: 隐藏层
        :return:
        """

        loss = 0
        with tf.GradientTape() as tape:
            # 1. 构建encoder inp?
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            # 2. 复制
            dec_hidden = enc_hidden
            # 3. <START> * BATCH_SIZE  BATCH_SIZE*1* self.embedding_dim ?
            dec_input = tf.expand_dims([self.vocab['<START>']] * self.BATCH_SIZE, 1)
            realBatchCount = 0
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # decoder(x, hidden, enc_output)
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

                tmpLoss = self.loss_function(targ[:, t], predictions)
                if tmpLoss !=  float('inf'):
                    loss += tmpLoss
                    realBatchCount += 1
                    #print('tmpLoss = {} , loss = {} '.format(tmpLoss, loss))
                else:
                    print('tmpLoss = {}'.format(tmpLoss,))
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

            #batch_loss = (loss / int(targ.shape[1]))
            batch_loss = (loss / realBatchCount)
            #取出encoder和decoder中的变量参数
            variables =  self.encoder.trainable_variables +  self.decoder.trainable_variables
            #计算梯度，更新权重
            gradients = tape.gradient(loss, variables)
            #用优化器更新
            self.optimizer.apply_gradients(zip(gradients, variables))
            bug = batch_loss.numpy()
            return batch_loss

    def beam_decode(self,batchData,beam_size = 2,min_dec_steps=2):
            """
            用beamsearch方法预测
            :param self:
            :param beam_size:beamwidth
            :param max_dec_steps:输出最大长度
            :param min_dec_steps:输出最小长度
            :return:
            """
            max_dec_steps = self.max_length_targ
            # 初始化mask
            start_index = vocab['<START>']
            stop_index = vocab['<STOP>']

            # 单步decoder
            def decoder_onestep(enc_output, dec_input, dec_hidden):
                # 单个时间步 运行
                preds, dec_hidden, attention_weights = self.call_decoder_onestep(dec_input, dec_hidden,
                                                                                                  enc_output)
                # 拿到top k个index 和 概率
                top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(preds), k=beam_size)
                # 计算log概率
                top_k_log_probs = tf.math.log(top_k_probs)
                # 返回需要保存的中间结果和概率
                return preds, dec_hidden, attention_weights, top_k_log_probs, top_k_ids

            # 计算第encoder的输出
            enc_output, enc_hidden = self.call_encoder(batchData)

            # 初始化batch size个 假设对象
            hyps = [Hypothesis(tokens=[start_index],
                               log_probs=[0.0],
                               hidden=enc_hidden[0],
                               attn_dists=[],
                               ) for _ in range( self.BATCH_SIZE)]
            # 初始化结果集
            results = []  # list to hold the top beam_size hypothesises
            # 遍历步数
            steps = 0  # initial step

            # 第一个decoder输入 开始标签
            dec_input = tf.expand_dims([start_index] *  self.BATCH_SIZE, 1)
            # 第一个隐藏层输入
            dec_hidden = enc_hidden

            # 长度还不够 并且 结果还不够 继续搜索
            while steps < max_dec_steps and len(results) < beam_size:
                # 获取最新待使用的token
                latest_tokens = [h.latest_token for h in hyps]
                # 获取所以隐藏层状态
                hiddens = [h.hidden for h in hyps]
                # 单步运行decoder 计算需要的值
                preds, dec_hidden, attention_weights, top_k_log_probs, top_k_ids = decoder_onestep(
                    enc_output,
                    dec_input,
                    dec_hidden)

                # 现阶段全部可能情况
                all_hyps = []
                # 原有的可能情况数量
                num_orig_hyps = 1 if steps == 0 else len(hyps)

                # 遍历添加所有可能结果
                for i in range(num_orig_hyps):
                    h, new_hidden, attn_dist = hyps[i], dec_hidden[i], attention_weights[i]
                    # 分裂 添加 beam size 种可能性
                    for j in range(beam_size):
                        # 构造可能的情况
                        new_hyp = h.extend(token=top_k_ids[i, j].numpy(),
                                           log_prob=top_k_log_probs[i, j],
                                           hidden=new_hidden,
                                           attn_dist=attn_dist)
                        # 添加可能情况
                        all_hyps.append(new_hyp)

                # 重置
                hyps = []
                # 按照概率来排序
                sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)

                # 筛选top前beam_size句话
                for h in sorted_hyps:
                    if h.latest_token == stop_index:
                        # 长度符合预期,遇到句尾,添加到结果集
                        if steps >= min_dec_steps:
                            results.append(h)
                    else:
                        # 未到结束 ,添加到假设集
                        hyps.append(h)

                    # 如果假设句子正好等于beam_size 或者结果集正好等于beam_size 就不在添加
                    if len(hyps) ==  self.BATCH_SIZE or len(results) ==  self.BATCH_SIZE:
                        break

                steps += 1

            if len(results) == 0:
                results = hyps

            hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
            results = []
            for best_hyp in hyps_sorted:
                best_hyp.abstract = " ".join([self.reverse_vocab[index] for index in best_hyp.tokens])
                print(best_hyp.abstract)
                results.append(best_hyp.abstract)
            #best_hyp = hyps_sorted[0]
            return best_hyp

    def greedy_evaluate(self,sentence):
            """
            利用贪婪推理算法进行的预测
            :param sentence:
            :return:
            """
            attention_plot = np.zeros((self.max_length_targ, self.max_length_inp + 2))

            inputs = pad_proc(sentence,self. max_length_inp, vocab)

            inputs = transform_data(inputs, self.vocab)

            inputs = tf.convert_to_tensor(np.array([inputs]))

            result = ''

            hidden = tf.zeros((1, self.units))
            enc_out, enc_hidden = self.encoder(inputs, hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([vocab['<START>']], 0)

            for t in range(self.max_length_targ):
                predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                     dec_hidden,
                                                                     enc_out)

                # storing the attention weights to plot later on
                attention_weights = tf.reshape(attention_weights, (-1,))

                attention_plot[t] = attention_weights.numpy()
                predicted_id = tf.argmax(predictions[0]).numpy()

                result += self.reverse_vocab[predicted_id] + ' '
                if self.reverse_vocab[predicted_id] == '<STOP>':
                    return result, sentence, attention_plot

                # the predicted ID is fed back into the model
                dec_input = tf.expand_dims([predicted_id], 0)

            return result, sentence, attention_plot

    def loss_function(self,real, pred,):
        """
        损失函数
        :param real:真实值label
        :param pred:预测值
        :return:
        """
        # 判断logit为1和0的数量,计算出<PAD>的数量有多少，并在mask中标记为0
        mask = tf.math.logical_not(tf.math.equal(real,  self.pad_index))
        # 计算decoder的长度，除去<PAD>字符数
        dec_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1)
        if dec_lens == 0:
            return float('inf')

        # 计算loss值
        loss_ = self.loss_object(real, pred)
        # 转换mask的格式
        mask = tf.cast(mask, dtype=loss_.dtype)
        # 调整loss，将<PAD>的loss归0
        loss_ *= mask
        # 确认下是否有空的摘要别加入计算，每一行累加
        loss_ = tf.reduce_sum(loss_, axis=-1) / dec_lens

        return loss_ #tf.reduce_mean(loss_)


    def test_evaluate(self,output_file):

        results = []#[{'QID':'1','Prediction':'SSSS'},{'QID':'2','Prediction':'SssSSS'}]
        for (batch, (inp)) in enumerate(self.dataset.take(self.steps_per_epoch)):
            result = self.beam_decode(inp)

            results += result
        df = pd.DataFrame(results,columns=['QID,Prediction'])
        df.to_csv(output_file,index=None)





if __name__ == '__main__':
    setproctitle.setproctitle('kelly')
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'AutoMaster' + os.sep
    print(dir)

    '''
    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')
    p = [0, 1, 2,0]
    p = np.array(p)
    p1 =  [
        [0.9, 0.05, 0.05], [-0.5, 0.89, 0.6], [0.05, 0.01, 0.94], [0.05, 0.01, 0.94]
    ]
    p1 = np.array(p1)
    loss = cce(p,p1)
    print('Loss: ', loss.numpy())  # Loss: 0.3239
    '''
    #训练数据
    embeddingModelFile = dir + 'fasttext/fasttext_jieba.model'
    vocab,reverse_vocab, embedding_matrix = getEmbedding_matrixFromModel(embeddingModelFile)
    train_x_pad_path = dir + 'AutoMaster_Train_X.csv'
    train_y_pad_path = dir + 'AutoMaster_Train_Y.csv'
    test_x_pad_path = dir + 'AutoMaster_Test_X.csv'
    train_X = load_dataset(train_x_pad_path,vocab)
    train_Y = load_dataset(train_y_pad_path,vocab)


    modelFile = dir+'checkpoints' + os.sep
    '''
    seq2seq = Seq2seq(type=Seq2seq.TYPE.TRAIN, train_X = train_X,train_Y = train_Y,vocab = vocab, embedding_matrix= embedding_matrix,modelFile=modelFile)
    seq2seq.train(True)
    
    #用greedy预测数据
    sentence = '我 的 帕萨特 烧 机油 怎么办 怎么办 技师 说 你好 请问 你 的 车 跑 了 多少 公里 了 如果 在 保修期 内 可以 到 当地 的 4 店 里面 进行 检查 维修 如果 已经 超出 了 保修期 建议 你 到 当地 的 大型 维修 店 进行 检查 烧 机油 一般 是 发动机 活塞环 间隙 过大 和 气门 油封 老化 引起 的 如果 每 750 0 公里 烧一升 机油 的话 可以 在 后备箱 备 一些 机油 以便 机油 报警 时有 机油 及时 补充 如果 超过 两升 或者 两升 以上 建议 你 进行 发动机 检查 维修 技师 说 你好 车主 说 嗯'
    s2s = Seq2seq(type=Seq2seq.TYPE.EVAL, vocab=vocab,
                      reverse_vocab=reverse_vocab, embedding_matrix=embedding_matrix, modelFile=modelFile)
    #result, _, attention_plot = s2s.greedy_evaluate(sentence)
    translate(sentence, 200, vocab, s2s.greedy_evaluate)
    '''
    #用beamsearch预测数据
    test_X = load_dataset(test_x_pad_path,vocab)
    s2s = Seq2seq(type=Seq2seq.TYPE.TEST, test_X=test_X, vocab=vocab,
                      reverse_vocab=reverse_vocab, embedding_matrix=embedding_matrix, modelFile=modelFile)
    output_file =  dir + 'AutoMaster_Test_Prediction.csv'
    s2s.test_evaluate(output_file)



