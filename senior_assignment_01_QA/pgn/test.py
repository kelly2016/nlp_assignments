# -*- coding:utf-8 -*-
# Created by LuoJie at 12/12/19
import tensorflow as tf

from pgn.batcher import batcher
from pgn.pgn import PGN
from pgn.test_helper import beam_decode
from pgn.utils.config import checkpoint_dir
from pgn.utils.params_utils import get_params
from pgn.utils.wv_loader import Vocab


def test(params):
    #assert params["mode"].lower() == "test", "change training mode to 'test' or 'eval'"
    #assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
    setproctitle.setproctitle('kelly')
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'AutoMaster' + os.sep
    print(dir)

    print("Building the model ...")
    model = PGN(params)

    print("Creating the vocab ...")
    vocab_path
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the batcher ...")
    b = batcher.batcher(vocab, params)

    print("Creating the checkpoint manager")
    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")

    for batch in b:
        print(beam_decode(model, batch, vocab, params))


def test_and_save(params):
    assert params["test_save_dir"], "provide a dir where to save the results"
    gen = test(params)
    with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
        for i in range(params["num_to_test"]):
            trial = next(gen)
            with open(params["test_save_dir"] + "/article_" + str(i) + ".txt", "w") as f:
                f.write("article:\n")
                f.write(trial.text)
                f.write("\n\nabstract:\n")
                f.write(trial.abstract)
            pbar.update(1)


if __name__ == '__main__':
    # 获得参数
    params = get_params()

    params["batch_size"] = 3
    params["mode"] = "test"
    test(params)
