# -*- coding: utf-8 -*-
# @Time    : 2019-04-24 11:49
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : bert_client.py
# @Description:
import setproctitle

from flask import Flask

CSRF_ENABLED = True
SECRET_KEY = 'you-will-never-guess'
app = Flask(__name__, instance_path='/Users/henry/Documents/application/newsExtract/instance/folder')
app.config['SECRET_KEY'] = SECRET_KEY

from bert_serving.client import BertClient
import  time
IP ='127.0.0.1' #
def class_pred(list_text):
    #文本拆分成句子
    print("total setance: %d" % (len(list_text)) )
    with BertClient(ip=IP, port=12544,port_out=12546, timeout=15000 ) as bc: # show_server_config=False, check_version=False,, check_length=False
        start_t = time.perf_counter()
        rst = bc.encode(list_text)
        print('result:', rst)
        print('time used:{}'.format(time.perf_counter() - start_t))
    #返回结构为：
    # rst: [{'pred_label': ['0', '1', '0'], 'score': [0.9983683228492737, 0.9988993406295776, 0.9997349381446838]}]
    #抽取出标注结果
    #pred_label = rst[0]["pred_label"]
    #result_txt = [ [pred_label[i],list_text[i] ] for i in range(len(pred_label))]
    #return result_txt


if __name__ == '__main__':
    setproctitle.setproctitle('bertshow')
    list_text = ['你好','我喜欢数学和语文']
    class_pred(list_text)
    app.run(
        host='0.0.0.0',
        port=9990,
        debug=True#Flask配置文件在开发环境中，在生产线上的代码是绝对不允许使用debug模式，正确的做法应该写在配置文件中，这样我们只需要更改配置文件即可但是你每次修改代码后都要手动重启它。这样并不够优雅，而且 Flask 可以做到更好。如果你启用了调试支持，服务器会在代码修改后自动重新载入，并在发生错误时提供一个相当有用的调试器。
    )
