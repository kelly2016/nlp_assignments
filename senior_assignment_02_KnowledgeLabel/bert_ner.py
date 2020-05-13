# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import csv
import os
import re

import tensorflow as tf

from birnn_crf_layer import BIRNN_CRF, CELL_TYPE
from model.bert import modeling
from model.bert import optimization
from model.bert import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

SPACE = ' '

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class NER_Processor(DataProcessor):
    """Processor for the Baidu_95 data set"""

    def __init__(self):
        self.language = "zh"

    '''
    @staticmethod
    def loadExamples(file):
        examples = []
        with open(file, encoding='utf-8') as fr:
            lines = fr.readlines()
            sent_, tag_ = [], []
            for (i, line) in enumerate(lines):
                if line != '\n':
                    tmp = line.strip().split('\t')
                    if (len(tmp) > 1):
                        guid = "train-%d" % (i)

                        text_a = tokenization.convert_to_unicode(re.sub(r'[0-9]+$', "", tmp[0]))
                        label = tokenization.convert_to_unicode(tmp[1])
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
                    else:
                        print('warning:the {} example is {}'.format(i, line))
                else:
                    print('2 warning:the {} example is {}'.format(i, line))

        return examples
    '''

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            # if i == 0:
            #     print('label: ', label)
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    @staticmethod
    def loadExamples(file):
        examples = []
        with open(file, encoding='utf-8') as fr:
            words = []
            labels = []
            guid = 0
            lines = fr.readlines()
            for (i, line) in enumerate(lines):
                if line == '\n':#一句话结束
                    assert len(words) == len(labels)
                    if len(words) < 1:#空格
                        continue

                    examples.append(
                        InputExample(guid= "train-%d" % (guid), text_a=SPACE.join(words), text_b=None, label=SPACE.join(labels)))
                    guid += 1
                    words = []
                    labels = []

                else:
                    tmp = line.strip().split('\t')
                    if (len(tmp) > 1):
                        words.append(tokenization.convert_to_unicode(re.sub(r'[0-9]+$', "", tmp[0])))
                        labels.append(tokenization.convert_to_unicode(tmp[1]))

                    else:
                        print('warning:the {} example is {}'.format(i, line))



        return examples


    def get_train_examples(self, data_dir):
        return self.loadExamples(os.path.join(data_dir, "weiboNER_2nd_conll.train"))

    def get_dev_examples(self, data_dir):
        return self.loadExamples(os.path.join(data_dir, "weiboNER_2nd_conll.test"))

    def get_test_examples(self, data_dir):
        return self.loadExamples(os.path.join(data_dir, "weiboNER_2nd_conll.dev"
                                                        ""))

    def get_labels(self):
        """

        :return:
        """
        return ['O','I-GPE.NOM','I-GPE.NAM','B-GPE.NOM','B-GPE.NAM','B-PER.NAM','I-PER.NAM','B-PER.NOM','I-PER.NOM','B-LOC.NAM','I-LOC.NAM','B-LOC.NOM','I-LOC.NOM','B-GPE.NAM','I-GPE.NAM','B-ORG.NAM','I-ORG.NAM','B-ORG.NOM','I-ORG.NOM','[CLS]', '[SEP]']


class KnowledgeLabel_Processor(DataProcessor):
    """Processor for the Baidu_95 data set"""

    def __init__(self):
        self.language = "zh"

    @staticmethod
    def load_examples(data_dir):
        with open(os.path.join(data_dir, "token_in.txt"), encoding='utf-8') as token_in_f:
            token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
            with open(os.path.join(data_dir, "predicate_single_out.txt"), encoding='utf-8') as predicate_out_f:
                #token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
                predicate_label_list = [seq.replace("\n", '') for seq in predicate_out_f.readlines()]
                assert len(token_in_list) == len(predicate_label_list)
                examples = list(zip(token_in_list, predicate_label_list))
                return examples

    def get_train_examples(self, data_dir):
        return self.create_example(self.load_examples(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self.create_example(self.load_examples(os.path.join(data_dir, "valid")), "valid")

    def get_test_examples(self, data_dir):
        return self.create_example(self.load_examples(os.path.join(data_dir, "test")), "test")

    def get_labels(self):
        """
         20 labels and  953 labels
        ['植物体细胞杂交的过程', '法国共和制的确立', '生物工程技术药物与疫苗的生产原理', '世界洋流分布规律及其对地理环境的影响', '民事权利和义务', '公民参与民主决策的多种方式', '自然资源的分布和利用', '固定化酶和固定化细胞', '人口迁移与人口流动', '人类对宇宙的新探索', '伟大的历史开端', '基因诊断和基因治疗', '洋务派近代工业的创办及其影响', '秦始皇', '人民民主的广泛性和真实性', '生物模型的种类与建构', '无机盐的主要存在形式和作用', '区域可持续发展', '水体的运动规律', '生态恢复工程', '内环境的理化特性', '人口分布与人口合理容量', '植物病虫害的防治原理和技术', '民族精神的时代特征', '制成固相化乳糖酶并检测牛奶中乳糖的分解', '中国的天气和气候', '实践是认识的动力', '诱发基因突变的因素', '现代生物技术专题', '植物组织培养的概念及原理', '公司经营与公司发展', '昆曲', '选择合理的旅游路线', '植物培养的条件及过程', '细胞的多样性和统一性', '免疫功能异常与免疫应用', '城乡建设与生活环境', '基因工程的概念', '消费类型', '胚胎干细胞的研究与应用', '城乡规划', '20世纪世界音乐的发展变化', '第三产业的兴起和“新经济”的出现', 'DNA分子的结构特征', '细胞膜的制备方法', '实事求是', '文化创新的源泉和动力', '脂肪酶、蛋白酶的洗涤效果', '外力作用与地貌', '遥感(RS)在资源普查、环境和灾害监测中的应用', '地理信息系统(GIS)在城市管理中的应用', '地球所处的宇宙环境', '动荡中变化的近代社会生活', '区域工业化和城市化的推进过程产生的问题及解决措施', '对外开放格局的初步形成', 'nan', '影响光合作用速率的环境因素及实践应用', '酶的发现历程', '中国的西北地区和青藏地区', '不同区域自然环境、人类活动的差异', '微生物的分离', '不断完善中国共产党的领导方式和执政方式', '物质循环', '大气运动', '国民经济的劫难', '酶的概念', '我国政府的宗旨和政府工作的基本原则', '细胞的无丝分裂', '贯彻新发展理念', '遗传的分子基础', '古典经济学巨匠的理论遗产', '执政为民', '基因对性状的控制', '内质网的结构和功能', '光反应和暗反应的比较', '水的存在形式及生理功能', '物种的概念及形成', '新思想的萌发', '对环境的伦理关怀', '大气受热过程', '英国君主立宪制的建立', '实践是认识的目的', '核糖体的结构和功能', '英国国会与国王的殊死搏斗', '同源染色体与非同源染色体的区别与联系', '马克思主义中国化的理论成果', '“斯大林模式”', '政府的职能：管理与服务', '细胞融合的方法', '商业中心及其形成', '用土壤浸出液进行细菌培养', '资本主义世界经济体系的形成', '个体衰老与细胞衰老', '探究培养液中酵母种群数量的动态变化', '人民群众创造历史的作用', '酶的特性', '动物细胞核移植技术', '测交方法检验F1的基因型', '教育的复兴', '国共合作与北伐战争', '探索生命起源之谜', '第二次世界大战爆发的历史背景', '光合作用的实质和意义', '意识能动性的表现', '生物的多样性', '议会权力的确立', '新文化运动', '美苏冷战局面的形成', '种群数量的变化', '主要农业地域类型的特点及其形成条件', '社会主义社会的基本矛盾的特点', '对分离现象的解释和验证', '曲折的发展', '开创中国特色社会主义的新篇章', '世界的居民和政区', '细胞的有丝分裂', '柏拉图和亚里士多德', '用DNA片段进行PCR扩增', '生物变异的应用', '地表形态对聚落及交通线路分布的影响', '全球气候变化对人类活动的影响', '马克思主义经济学的伟大贡献', '基因工程的应用', '蛋白质分子的化学结构和空间结构', '配子形成过程中染色体组合的多样性', '社会主义市场经济的基本特征', '免疫系统的组成', '意大利资本主义萌芽', '液泡的结构和功能', '自然地理环境整体性的表现', '细胞周期的概念', '生物武器对人类的威胁', '中国的振兴', '宇宙中的地球', '细胞器之间的协调配合', '收入分配方式对效率、公平的影响', '单克隆抗体的应用', '影响细胞呼吸的因素', '性状分离比的模拟实验', '细胞呼吸原理在生产和生活中的应用', '果胶酶的活性测定', '精子和卵细胞形成过程的异同', '培养基与无菌技术', '海洋和海岸带', '人体神经调节的结构基础和调节过程', '政府的责任：对人民负责', '选官、用官制度的变化', '东欧剧变与苏联解体', '新时代我国社会主要矛盾', '量子论的诞生与发展', '如何进行生态工程的设计', '孔子', '真核细胞的三维结构模型', '改良蒸汽机', '农林牧副鱼一体化生态工程', '动乱中的教育', '传统的发酵技术', '经济学常识', '高中', '董仲舒的“罢黜百家', '五四运动', '结合实践', '生物膜的流动镶嵌模型', '多种所有制经济共同发展', '制备和应用固相化酶', '西方国家现代市场经济的兴起与主要模式', '基因是有遗传效应的DNA片段', '蛋白质的合成', '脂质的组成元素', '世界贸易组织和中国的加入', '意识的内容与形式', '遗传物质的探寻', '传统文化的特点及其影响', '地壳内部物质循环过程', '内环境维持稳态的生理机制', '克伦威尔', '叶绿体色素的提取和分离', '植物激素及其植物生长调节剂的应用价值', '细胞质壁分离与质壁分离复原现象及其原因', '世界的陆地和海洋', '生活中的法律常识', '免疫调节', '内力作用与地貌', '中美关系正常化和中日邦交正常化', '治疗性克隆的操作过程', '文化建设的必然要求', '三星堆遗址', '14年时事热点', '民族区域自治制度的建立', '《伤寒杂病论》和《本草纲目》', '树立科学思维的观念', '价值的实现方式', '水土流失的治理', '劳动就业与守法经营', '从生物材料中提取某些特定成分', '现代生态工程的农业模式', '伴性遗传', '人体的体温调节', '致癌因子与癌变', '中国的自然资源', '气候类型及其判断', '自然地理要素的相互作用', '酶的存在和简单制作方法', '噬菌体侵染细胞的实验', '中央官制——三公九卿制', '检测蛋白质的实验', '亲子代生物之间染色体数目保持稳定的原因', '我国的人口现状与前景', '中外著名旅游景区的景观特点及其形成原因', '确定旅游点', '我国的民族区域自治制度', '世界主要陆地自然带', '用比色法测定亚硝酸盐含量的变化', '基因与性状的关系', '无氧呼吸的类型', '人口增长与人口问题', '生物膜的结构的探究历程', '等压面、等压线、等温线等分布图的判读', '染色体变异', '生长素类似物在农业生产实践中的作用', '神经调节与体液调节的关系', '生产活动中地域联系的重要性和主要方式', '生态系统的成分', '大洋洲和南极洲', '遗传与进化', '生物的性状与相对性状', '动、植物细胞有丝分裂的异同点', '核酸在生命活动中的作用', '不同发展阶段地理环境对人类生产和生活方式的影响', '基因工程', '脂质的种类及其功能', '电气技术的应用', '环境污染形成的原因、过程及危害', '加强思想道德建设', '遵循形式逻辑的要求', '和平与动荡并存', '水循环过程和主要环节及地理意义', '实施“新政”', '“文化大革命”', '临危受命', '三大产业发展及其产业结构转移', '市场调节及其弊端', '反射的过程', '杂交育种', '流域开发的地理条件、内容、综合治理措施', '细胞凋亡的概念', '与细胞分裂有关的细胞器', '宪章运动', '新世纪新阶段中国共产党的旗帜', '信息的种类与传递', '皇帝制度', '基因连锁和交换定律', '生活在社会主义法治国家', '宰相制度的废除与内阁的出现', '明治维新', '蛋白质的提取和分离', '南书房、军机处的设立', '文艺复兴', '文化与综合国力', '中国特色的政党制度', '胞吞、胞吐的过程和意义', '内环境的稳态', '抗生素的合理使用', '碳原子的结构特点', '地球的形状和大小', '宪法对我国国家性质的规定', '“一国两制”的理论和实践', '自然选择对种群基因频率变化的影响', '日益重要的国际组织', '细胞工程的概念', '社会意识的相对独立性', '中华人民共和国的成立', '生物技术研究与开发的有关政策和法规', '发展社会主义民主政治', '理性时代的到来', '动物细胞工程的常用技术', '细胞核的功能', '脱氧核苷酸序列与遗传信息的关系', '地球是太阳系中一颗既普通又特殊的行星', '胚胎分割移植', '祖国统一的历史潮流', '民主政治的重要文献', '利用神经调节原理的仿生学', '七八十年代美苏由紧张对抗到谋求缓和对话', '影响文化发展的主要因素', '第一次世界大战的过程', '西欧国家的殖民扩张与争夺', '明清小说', '生态系统的概念与内涵', '《人权宣言》', '植物色素的提取', '中东战争', '现代文化传播手段的特点', '粮食问题', '法律救济', '单克隆抗体的制备', '保护生物多样性的措施和意义', '探索与失误', '社会主义核心价值观的基本内容', '细胞学说建立过程', '孔子和早期儒学', '蛋白质的盐析与变性', '现实主义美术和印象画派', '第二次鸦片战争的概况及影响', '中国各民族对中华文化的贡献', '制作腐乳的科学原理及影响腐乳品质的条件', '细胞呼吸的概念', '可持续发展的基本内涵及协调人地关系的主要途径', '历史与历史的重现', '建设现代化经济体系', '观察植物细胞的有丝分裂实验', '发展的概念', '主动运输的原理和意义', '20世纪的世界文学', '矛盾的普遍性和特殊性', '土壤中分解尿素的细菌的分离与计数', '尿糖的检测', '地球仪', '监察制度', '启蒙运动的扩展', '水运与航空', '细胞有丝分裂不同时期的特点', '真核细胞的分裂方式', '世界主要自然灾害带的分布', '生物工程技术', '地球运动的地理意义', '十月革命一声炮响', '人类活动对群落演替的影响', '夏商两代的政治制度', '减数分裂过程中染色体和DNA的规律性变化', '脂质在生命活动中的作用', '等值线图', '对外开放', '用组织培养法培养花卉等幼苗', '细胞膜的结构特点', '步入世界外交舞台', '设施农业', '生物技术中的伦理问题', '政治', '凡尔赛—华盛顿体系下的和平', '地球的内部圈层结构及特点', '官营手工业、民营手工业、家庭手工业的消长', '核酸的种类以及在细胞中的分布', '生产决定消费', '区域的含义', '世界文化与民族文化的关系', '经济基础与上层建筑的相互作用及其矛盾运动', '细胞中糖类的种类、分布和功能', '中华民族精神的核心', '细胞衰老、凋亡、坏死与癌变的比较', '中国恢复在联合国合法席位', '纯化大肠杆菌的无菌操作和菌种保存', '土壤中小动物类群丰富度的研究', '社会主义从理想到现实的转变', '证明DNA是主要遗传物质的实验', '中国走可持续发展道路的必然性', '氨基酸的分子结构特点和通式', '交通运输与地理环境', '渗透系统和渗透作用', '海峡两岸关系的发展', '中华文化的包容性', '碱基互补配对原则', '胚胎工程的概念及操作对象', '世界经济区域集团化', '叶绿体的结构和功能', '农业区位因素', '彰显文化自信的力量', '原核细胞和真核细胞的形态和结构的异同', '生态农业工程', '铁路与公路', '中华民族精神的基本内涵', '血糖平衡的调节', '奥斯威辛集中营', '两极对峙格局的形成', '人类遗传病的监测和预防', 'RNA分子的组成和种类', '自然灾害发生的主要原因及危害', 'DNA的分子组成', '观察蝗虫精母细胞减数分裂固定装片实验', '中国共产党的性质、宗旨和指导思想', '酶的作用和本质', '中国台湾省、香港和澳门', '联合国', '细胞大小与物质运输的关系', '细胞免疫的概念和过程', '国有经济及其主导作用', '生物膜的功能特性', '光合作用的探究历程', '细胞工程', '旅游景观的欣赏', '科学社会主义常识', '孙中山', '高倍镜观察叶绿体和线粒体', '如何看待落后文化和腐朽文化', '人民代表大会及其常设机关的法律地位', '化能合成作用', '我国行政监督体系', '商业的发展', '遗传信息的转录和翻译', '德意志帝国君主立宪制的确立', '我国的村民自治与城市居民自治及其意义', '城市地域结构及服务功能', '公民直接参与民主决策的意义', '培养担当民族复兴大任的时代新人', '人类认识的宇宙', '线粒体的结构和功能', '发展的量变与质变状态', '社会主义市场经济的伦理要求', '单因子和双因子杂交实验', '文化对人影响的特点', '我国公民必须履行的政治义务', '面对科技进步的伦理冲突', '城乡分布', '中华民国的成立及《中华民国临时约法》的颁布', '生物净化的原理和方法', 'DNA的粗提取和鉴定', '对待传统文化的正确态度', '世界气候类型和自然景观', '全球性生态环境的影响', '文化与经济、政治的关系', '水资源的合理利用', '人民群众的概念', '垄断组织的出现', '古代希腊民主政治产生的历史条件', '细胞凋亡与细胞坏死的区别', '信息传递在生态系统中的作用及应用', '中国特色社会主义是由道路、理论体系、制度三位一体构成的', '生态系统的功能', 'DNA分子的复制', '有丝分裂的意义', '第一国际和第二国际', '生态环境问题的成因及其形成的一般过程', '汉字与书法艺术', '个人价值与社会价值的统一', '中国的疆域行政区划、人口和民族', '近代史', '拿破仑', '禁止生殖性克隆', '洛克', '马克思主义在中国的传播', '自然经济的逐步解体', '提取芳香油', '布雷顿森林体系的建立', '生态工程的建设情况综合', '雅典民主政治', '生态系统的营养结构', '细胞的衰老', '生物进化论', '《独立宣言》', '明清之际的儒学思想', '近代中国的思想解放潮流', '内环境的组成', '系统优化方法', '达尔文的自然选择学说', '湿地资源的开发与保护', '中华文化源远流长、博大精深', '溶酶体的结构和功能', '神经调节和体液调节的比较', '酶活力测定的一般原理和方法', '细胞的增殖', '毛泽东思想的形成', '“三个代表”重要思想', '生物体维持pH稳定的机制实验', '坚持社会主义核心价值体系', '生命活动离不开细胞', '矛盾的同一性和斗争性', '为人民服务的政府', '1929至1933年资本主义世界经济危机', '科学发展观', '染色质和染色体的结构', '邓小平理论', '中华民族复兴的必然选择', '工业区位因素', '植物组织培养技术的实践应用', '细胞膜具有流动性的原因', '艾滋病的流行和预防', '地域分异的基本规律', '分子与细胞', '探究植物生长调节剂对扦插枝条生根的作用', '细胞癌变及癌细胞的主要特征', '新发展理念和中国特色社会主义新时代的经济建设', '从“无为“到”有为”', '氨基酸及其种类', '人民代表大会制度的基本内容', '细胞呼吸', '精子的形成过程', '恶性肿瘤的防治', '独尊儒术”', '载人航天技术的发展', '时代的主题', '春秋战国时期的乐舞', '“重农抑商”政策', '影视艺术的产生与发展', '探究酵母菌的呼吸方式', '常见的天气系统', '虚假“繁荣”的幻灭', '制作泡菜', '凯末尔', '当地自然群落中若干种生物的生态位', '消费对生产的反作用', '共产党领导的多党合作和政治协商制度的形成', '鸦片战争的概况及影响', '文化继承与发展的关系', '中国的交通运输业、商业和旅游业', '生物进化观念对人们思想观念的影响', '突触的结构', '酶在食品制造和洗涤等方面的应用', '人类迈入电气时代', '激素调节概念及特点', '蛋白质多样性的原因', '城市化的过程和特点', '细胞的失水和吸水', '生物技术实践', '19世纪法国政体的变迁', '重庆谈判和内战的爆发', '《资政新篇》及太平天国运动的进步性与局限性', '基因工程的原理及技术', '青少年中常见的免疫异常', '宏观调控', '世界文化多样性的表现', '免疫系统的功能', '公民道德与伦理常识', '发酵食品加工的基本方法及应用', '我国公民参与政治生活的基本原则和主要内容', '地球与地图', '细胞的衰老和凋亡与人类健康的关系', '抗日战争', '各具特色的国家和国际组织', '区域经济与经济重心的南移', '“百花齐放、百家争鸣”的方针', '通讯工具的变迁', '善于创新', '毛泽东思想的发展', '维新思想', '糖类的种类和作用的综合', '中国共产党的成立', '生物性污染', '区域农业生产的条件、布局特点、问题', '创新的社会作用', '体液免疫的概念和过程', '现代交通运输方式的特点和城市交通的特点', '脑的高级功能', '探究水族箱（或鱼缸）中群落的演替', '17年时事热点', '数字地球的含义', '八国联军侵华战争', '历史', '台风、寒潮、干旱、洪涝等气象灾害的形成原因', '创新与继承的关系', '动物胚胎发育的概念', '圣彼得大教堂', '华盛顿', '转基因生物和转基因食品的安全性', '群落的物种组成', '细胞膜的功能', '中国特色社会主义文化的基本内涵', '动植物细胞的主要区别', '微生物的培养与应用', '生长素的作用以及作用的两重性', '动物细胞核具有全能性的原因及其应用', '生物种间关系', '组成细胞的化学元素', '报刊业走向繁荣', '20世纪50年代至70年代探索社会主义建设道路的实践', '诱变育种', '基因重组概念及类型', '反射弧各部分组成及功能', '大众传媒的发展', '生态工程依据的生态学原理', '核膜和核孔的结构和功能', '避孕的原理和方法', '西周的宗法制、分封制与礼乐制度', '《天朝田亩制度》的颁布', '当代资本主义的新变化', '生态工程建设的基本过程', '群众观点和群众路线', '财政与宏观调整', '动物细胞与组织培养过程', '民主集中制', '神经冲动的含义', '我国政府权威的来源和树立', '巴黎公社', '辩证法的革命批判精神与创新意识', '区域能源、矿产资源的开发与区域可持续发展的关系', '多极化趋势在曲折中发展', '中共十一届三中全会', '微生物数量的测定', '生物', '西学为用”——洋务运动', '细胞中的脂质的种类和功能', '新时代发展目标', '蛋白质工程', '传统工业区和新兴工业区', '我国公民的民主监督权和实行民主监督的合法渠道', '激素的种类、功能与应用', '兴奋在神经元之间的传递', '光合作用的过程', '3S技术综合应用', '尊重文化多样性的意义', '细胞的分化及意义', '原核细胞与真核细胞的比较', '西亚和北非、撒哈拉以南非洲', '中国的地形', '基因工程的基本操作程序', '荒漠化的防治', '资本主义萌芽', '北魏孝文帝改革', '基因、蛋白质与性状的关系', '转基因生物的安全性问题', '“开眼看世界”', '新航路开辟的原因及条件', '太阳对地球的影响', '城市的空间结构及其形成原因', '法国的启蒙思想家', '香港、澳门的回归', '城市化对地理环境的影响', '人民代表大会的职权', '新民主主义革命的胜利及其伟大意义', '基因突变对蛋白质和形状的影响', '勃列日涅夫改革', '自由扩散和协助扩散', '和平共处五项原则', '文化传播的多种途径', '肺炎双球菌的转化实验', '劳动与就业', '胚胎移植', '创新与借鉴、融合的关系', '辛亥革命的评价', '马克思、恩格斯', '陆地上水体间的相互关系', '京剧的出现', '地球的外部圈层结构及特点', '生命系统的结构层次', '人类遗传病的类型及危害', '其他细胞器主要功能', 'ATP在能量代谢中的作用的综合', '19世纪的音乐流派与杰作', '生物科学与社会', '利用生物净化原理治理环境污染', '生物膜研究的前沿信息', '古代史', '植物细胞的全能性及应用', '清末民主革命风潮', '水和无机盐的作用的综合', '单倍体诱导与利用', '唯物辩证法的联系观', '一切从实际出发', '治污生态工程', '有氧呼吸的三个阶段', '检测脂肪的实验', '生物资源的合理利用', '细胞工程的操作水平、对象、条件及手段', '农业生产中繁殖控制技术', '欧洲西部、欧洲东部及北亚', '植物细胞壁的成分和功能', '相对论的创立', '现代主义美术', '唐太宗', '公民道德建设的内容与要求', '中国共产党领导和执政地位的确立', '家庭与婚姻', '生物氧化塘是如何净化污水的', '物质资料的生产方式是人类社会存在和发展的基础', '人工授精、试管婴儿等生殖技术', '动物体细胞克隆', '环境污染防治的主要措施', '意识能动性的特点', '拉马克的进化学说', '促进全面发展', '社会主义从空想到科学的发展', '发展中国特色社会主义文化', '植物体细胞杂交的应用', '东亚、东南亚、南亚和中亚', '细胞衰老的原因探究', '观察植物细胞的质壁分离和复原', '人类基因组计划及其意义', '动物疫病的控制', '社会主义经济理论', '种群的特征', '信守合同与违约', '不完全显性', '反法西斯战争胜利的历史意义', '等高线地形图、地形剖面图', '酒酵母制酒及乙酸菌由酒制醋', '运用辩证思维方法', '唯物辩证法的总特征', '生产力与生产关系的相互作用及其矛盾运动', '科学思维常识', '电子显微镜', '观察细胞的减数分裂实验', '高倍显微镜的使用', '欧洲共同体的成立', '细胞呼吸的过程和意义', '康熙帝', '碳循环与温室效应', '减数分裂与有丝分裂的比较', '种群数量的变化曲线', '细胞膜内外在各种状态下的电位情况', '组织培养基的成分及作用', '社会发展的实现方式', '唯物辩证法的实质与核心', '中国的农业和工业', '牛顿、爱因斯坦', '生态系统的反馈调节', '经济体制改革', '多极化趋势的加强', '糖类的作用', '基因重组特点及意义', '激素调节', '细胞观察实验', '两党制的形成和发展', '经纬网及其地理意义', '社会存在决定社会意识', '细胞膜的流动镶嵌模型', '克隆的利与弊', '生产果汁酒以及生产醋的装置', '时事政治', '凝魂聚气、强基固本的基础工程', '苏格拉底等', '培养基对微生物的选择作用', '农业的主要耕作方式和土地制度', '探究将有油渍、汗渍、血渍的衣物洗净的办法', '主要经济政策', '生物多样性形成的影响因素', '蛋白质在生命活动中的主要功能', '细胞融合的概念', '检测还原糖的实验', '卢梭', '植物或动物性状分离的杂交实验', '曲折的年代——十年动乱与文化凋零', '区域存在的环境与发展问题', '估算种群密度的方法', '丰富精神世界', '国际贸易和国际金融', '酶的特性在生产生活中的应用', '社会主义是中国人民的历史性选择', 'PCR技术的基本操作和应用', '现代史', '大抗议书', '低温诱导染色体加倍实验', '宋明理学', '革命道路的探索', '城市的区位因素', '中国民族资本主义的曲折发展', '转基因技术', '酶在工业生产中的应用', '18年时事热点', '罗马法的发展与完善', '“冷战”格局下的国际关系', '全球定位系统(GPS)在定位导航中的应用', '走进细胞', '转基因生物的利与弊', '现实主义文学', '人口性别构成、年龄构成图的判断', '生物大分子以碳链为骨架', '民主政治制度的建设', 'DNA分子的多样性和特异性', '神经、体液调节在维持稳态中的作用', '微生物的鉴定', '牢牢掌握意识形态工作领导权', '走向会合的世界——世界市场雏形出现', '民族资本主义的产生', '卵细胞的形成过程', '世界人民反战和平运动的高涨', '地震、泥石流、滑坡等地质地貌灾害的产生机制与发生过程', '基因突变的特点', '地理', '人体免疫系统在维持稳态中的作用', '文化的内涵与特点', '生长素的产生、分布和运输情况', '市场配置资源', '基因与DNA的关系', '探究细胞表面积与体积的关系', '三民主义的实践', '现代生物技术在育种上的应用', '联系的客观性', '探究膜的透性', '中心法则', '国民大革命', 'DNA与RNA的异同', '形而上学的否定观', '宗教改革', '整体和部分的关系', '组成细胞的化合物', '独立之初的严峻形势', '稳态与环境', '中国特色社会主义市场经济', '核酸', '经济全球化迅速发展及全球化存在的问题与展望', '我国的选举制度及选举方式', '细胞衰老的特征', '“两弹一星”', '细胞膜的成分', '短暂的春天', '经典力学', '动物胚胎发育的过程', '生殖技术的伦理问题', '细胞中的无机物', '建立“福利国家”', '大气、天气、气候变化规律', '国共政权的对峙', '甲午中日战争的概况及影响', '人口与城市', '核酸的基本组成单位', '“自由放任”政策的失败', '性状的显、隐性关系及基因型、表现型', '群落的结构', '对政府权力进行制约和监督的意义', '15年时事热点', '文化创新的意义', '从《诗经》到唐诗', '植物的组织培养', 'ATP与ADP的相互转化', '收集旅游信息', '产业转移和资源跨区域调配对区域地理环境的影响', '基因的自由组合规律的实质及应用', '生态工程的概念及其目标', '植物激素的概念', '交通运输布局及其变化对生产、生活和社会经济的影响', '有利于和不利于环境保护的消费行为', '遗传的细胞基础', '固相化酶的应用价值', '基因和遗传信息的关系', '生物技术在其他方面的应用', '交通运输线路及站点的区位因素', '中国的河流和湖泊', '染色体结构变异', '“中学为体', '地球运动的基本形式', '减数第一、二次分裂过程中染色体的行为变化', '提高政府依法行政的水平', '工业地域的形成条件与发展特点', '公有制为主体', '地壳物质循环及地表形态的塑造', '光照图的综合判读', '春秋战国时期的百家争鸣', '《辛丑条约》的签订', '中心体的结构和功能', '抗战时期和解放战争时期的民族工业', '细胞不能无限长大的原因', '种群和群落', '高尔基体的结构和功能', '中国东部季风区（北方地区和南方地区）', 'DNA、RNA在细胞中的分布', '微生物的应用', '互联网的兴起', '人工湿地的作用', '政府依法行政的意义和要求', '明清君主专制制度的加强', '戊戌变法', '地图的基本知识', '生产活动与地域联系', '战时共产主义政策和新经济政策', '摆脱危机困境', '基因的分离规律的实质及应用', '染色体数目变异', '减数分裂的概念', '文化对人影响的表现', '植物激素的作用', '商鞅变法', '大津巴布韦遗址与非洲文明探秘', '利用微生物进行发酵来生产特定的产物', '海洋开发', '北美和拉丁美洲', '常见的人类遗传病', '从新古典主义美术到浪漫主义美术', '伴性遗传在实践中的应用', '从珍妮机到蒸汽机', '核酸的结构组成和功能', '我国公民享有的政治权利', '探究影响酶活性的因素', '16年时事热点', 'DNA是主要的遗传物质', '绿色食品的生产', '马克思主义哲学的基本特征', '能量流动的特点与意义', '新文化运动与马克思主义的传播', '资本主义在中国近代历史发展进程中的地位和作用', '全球气压带、风带的分布、移动规律及其对气候的影响', '弘扬和培育民族精神的途径和意义', '器官移植', '法国大革命', '我国政府的主要职能', '马克思主义普遍原理与中国实际相结合', '坚持文化创新的正确方向', '民族资本主义的产生和初步发展', '胚胎干细胞的来源及分化', '干细胞的研究进展和应用', '推动文化交流的意义', '世界市场的拓展', '微生物的培养', '戈尔巴乔夫改革', '市场经济', '农业生态系统中的能量流动', '公民的道德生活', '影响(均衡)价格的因素', '染色体组的概念、单倍体、二倍体、多倍体', '人体水盐平衡调节', '生产技术的不断进步', '有丝分裂中结构和物质的变化', '英国责任内阁制的形成', '智者学派', '推动社会主义文化繁荣兴盛', '赫鲁晓夫改革', '生物技术在食品加工中的应用', '文学的主要成就', '植物生长素的发现和作用', '人民代表大会与其他国家机关的关系', '宋词和元曲', '文艺的春天', '革命前的沙皇俄国', '兴奋在神经纤维上的传导', '三民主义的提出', '神经元各部分的结构和功能', '责任制内阁的形成', '不同生物遗传物质的判定', '共产党宣言的问世', '试管婴儿及其意义', '古代中国政治制度的特点', '无氧呼吸的概念与过程', '郡县制', '改革开放以来民主与法制的建设', '地域文化对人口或城市的影响', 'ATP在生命活动中的作用和意义', '基因指导蛋白质的合成', '孟德尔遗传实验', '从习惯法到成文法', '糖类的组成元素', '细胞膜', '叶绿体结构及色素的分布和作用', '受精作用的概念和过程', '市场秩序', '程朱理学', '文化的社会作用', '第一次世界大战爆发的历史背景', '价值观的导向作用', '发酵与食品生产', '生物膜系统的功能', '文化市场对文化的影响', '胚胎干细胞的特点和种类', '国际经济合作', '复等位基因', '计算机和生物技术的发展', 'ATP的分子结构和特点', '中国共产党：以人为本', '辩证否定', '社会主义市场经济体制的建立', '“百家争鸣”局面的出现', '能量流动的概念和过程', '旅游业的发展对社会、经济、文化的作用', '手工业成就', '1787年宪法的颁布', '内战']


        :return:
        """
        return  ['__label__高中_生物_生物技术实践', '__label__高中_历史_近代史', '__label__高中_政治_科学思维常识', '__label__高中_历史_古代史', '__label__高中_生物_生物科学与社会', '__label__高中_生物_稳态与环境', '__label__高中_生物_现代生物技术专题', '__label__高中_政治_经济学常识', '__label__高中_地理_人口与城市', '__label__高中_地理_区域可持续发展', '__label__高中_生物_分子与细胞', '__label__高中_地理_生产活动与地域联系', '__label__高中_地理_地球与地图', '__label__高中_政治_科学社会主义常识', '__label__高中_地理_宇宙中的地球', '__label__高中_政治_公民道德与伦理常识', '__label__高中_生物_遗传与进化', '__label__高中_政治_时事政治', '__label__高中_历史_现代史', '__label__高中_政治_生活中的法律常识']


    @staticmethod
    def create_example(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_str = line[0]
            predicate_label_str = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_str, text_b=None, label=predicate_label_str))
        return examples

class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

def getLabel_map(label_list):
    label2id = {}
    for (i, label) in enumerate(label_list):
        label2id[label] = i
        # --- save label2id.pkl ---
        # 在这里输出label2id.pkl , add by kelly
    id2label = {value: key for key, value in label2id.items()}

    # --- Add end ---
    return label2id,id2label

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map, _ = getLabel_map(label_list)

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id= label_map["[SEP]"],
        is_real_example=False)

  #label_map = {}
  #for (i, label) in enumerate(label_list):
    #label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  label_id = example.label.split(' ')

  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]
      label_id = label_id[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  label_ids = []

  tokens.append("[CLS]")
  segment_ids.append(0)
  label_ids.append(label_map["[CLS]"])
  for i, token in enumerate(tokens_a):
    tokens.append(token)
    segment_ids.append(0)
    label_ids.append(label_map[label_id[i]])
  tokens.append("[SEP]")
  segment_ids.append(0)
  label_ids.append(label_map["[SEP]"])
  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(label_ids) == max_seq_length


  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))


  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_ids,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_id)
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
#kelly modify
  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  #kelly add 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
  embedding = model.get_sequence_output()# model.get_pooled_output()
  max_seq_length = embedding.shape[1].value #hidden_size = output_layer.shape[-1].value
  # 算序列真实长度
  used = tf.sign(tf.abs(input_ids))#如果x < 0,则有 y = sign(x) = -1；如果x == 0,则有 0 或者tf.is_nan(x)；如果x > 0,则有1.
  lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
  bigru_crf =BIRNN_CRF(embedded_chars=embedding, cell_type=CELL_TYPE.GRU,num_labels=num_labels,
                       seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)

  return bigru_crf. add_birnn_crf_layer()


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, logits, trans, predictions) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      hook_dict = {}
      hook_dict['loss'] = total_loss
      hook_dict['global_steps'] = tf.train.get_or_create_global_step()
      logging_hook = tf.train.LoggingTensorHook(
          hook_dict, every_n_iter=10)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:

      #def metric_fn(per_example_loss, label_ids, logits, is_real_example):
      def metric_fn(label_ids, predictions):
        #predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions)#, weights=is_real_example)
        print('label_ids= {} ,predictions = {}'.format(label_ids,predictions))
        auc = tf.metrics.auc(labels=label_ids, predictions=predictions, weights=is_real_example)
        precision = tf.metrics.precision(labels=label_ids, predictions=predictions, weights=is_real_example)
        recall = tf.metrics.recall(labels=label_ids, predictions=predictions, weights=is_real_example)

        loss =  tf.metrics.mean_squared_error(labels=label_ids, predictions=predictions)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
            #'eval_auc': auc,
            'eval_precision': precision,
            'eval_recall': recall,
        }

      eval_metrics = metric_fn(label_ids, predictions)
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metrics
      )
      '''
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
      '''
    else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

      #output_spec = tf.contrib.tpu.TPUEstimatorSpec(x
       #   mode=mode,
        #  predictions={"probabilities": probabilities},
         # scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {

      "ner":NER_Processor
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    '''
    # kelly add
    tensors_to_log = {"train loss": "loss","train global_steps":"global_steps"}
    logging_hook = tf.estimator.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)

    estimator.train(input_fn=train_input_fn, hooks=[logging_hook], max_steps=num_train_steps)
    '''
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)
    print("1 predicate_predict.csv is written ")
    output_predict_file = os.path.join(FLAGS.output_dir, "predicate_predict.txt")
    _, id2label = getLabel_map(label_list)
    def result_to_pair(writer):
        for predict_line, prediction in zip(predict_examples, result):
            idx = 0
            line = ''
            line_token = str(predict_line.text_a).split(' ')
            label_token = str(predict_line.label).split(' ')
            len_seq = len(label_token)
            if len(line_token) != len(label_token):
                tf.logging.info(predict_line.text_a)
                tf.logging.info(predict_line.label)
                break
            for id in prediction:
                if idx >= len_seq:
                    break
                curr_labels = id2label[id]
                if curr_labels in ['[CLS]', '[SEP]']:
                    continue
                try:
                    line += line_token[idx] + ' ' + label_token[idx] + ' ' + curr_labels + '\n'
                except Exception as e:
                    tf.logging.info(e)
                    tf.logging.info(predict_line.text_a)
                    tf.logging.info(predict_line.label)
                    line = ''
                    break
                idx += 1
            writer.write(line + '\n')

    with   codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
        result_to_pair(writer)
    from utils.conlleval import return_report

    eval_result = return_report(output_predict_file)
    print(''.join(eval_result))
    # 写结果到文件中
    with  codecs.open(os.path.join(FLAGS.output_dir, 'predict_score.txt'), 'a', encoding='utf-8') as fd:
        fd.write(''.join(eval_result))


if __name__ == "__main__":
  import setproctitle
  setproctitle.setproctitle('kelly')
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")

  tf.app.run()
