# -*- coding:utf-8 -*-
# Created by LuoJie at 12/7/19

import argparse


def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='train', help="run mode", type=str)

    parser.add_argument("--max_enc_len", default=400, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=41, help="Decoder input max sequence length", type=int)
    parser.add_argument("--batch_size", default=64, help="batch size", type=int)
    parser.add_argument("--epochs", default=10, help="train epochs", type=int)
    parser.add_argument("--vocab_path", default=vocab_path, help="vocab path", type=str)
    parser.add_argument("--learning_rate", default=1e5, help="Learning rate", type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. "
                             "Please refer to the Adagrad optimizer API documentation "
                             "on tensorflow site for more details.",
                        type=float)
    parser.add_argument("--max_grad_norm", default=0.8, help="Gradient norm above which gradients must be clipped",
                        type=float)

    parser.add_argument("--vocab_size", default=31820, help="max vocab size , None-> Max ", type=int)

    parser.add_argument("--beam_size", default=3,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)",
                        type=int)
    parser.add_argument("--embed_size", default=500, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=512, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=512, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=256, help="[context vector, decoder state, decoder input] feedforward \
                            result dimension - this result is used to compute the attention weights",
                        type=int)

    parser.add_argument("--train_seg_x_dir", default=train_x_seg_path, help="train_seg_x_dir", type=str)
    parser.add_argument("--train_seg_y_dir", default=train_y_seg_path, help="train_seg_y_dir", type=str)
    parser.add_argument("--test_seg_x_dir", default=test_x_seg_path, help="train_seg_x_dir", type=str)

    parser.add_argument("--checkpoints_save_steps", default=5, help="Save checkpoints every N steps", type=int)

    parser.add_argument("--min_dec_steps", default=4, help="min_dec_steps", type=int)


    parser.add_argument("--max_train_steps", default=1250, help="max_train_steps", type=int)
    parser.add_argument("--train_pickle_dir", default='/opt/kaikeba/dataset/', help="train_pickle_dir", type=str)
    parser.add_argument("--save_batch_train_data", default=False, help="save batch train data to pickle", type=bool)
    parser.add_argument("--load_batch_train_data", default=False, help="load batch train data from pickle", type=bool)

    args = parser.parse_args()
    params = vars(args)

    return params


def get_default_params(vocab, embedding_matrix):

    vocab_size = len(vocab)
    #x_max_len = 518 ,train_y_max_len = 46

    params = {'mode': 'train',
              'max_enc_len': 518,
              'max_dec_len': 46,
              'batch_size': 8,
              'epochs': 20,
              'learning_rate': 0.015,
              'adagrad_init_acc': 0.1,
              'max_grad_norm': 0.8,
              'vocab_size': vocab_size,
              'beam_size': 3,
              'embed_size': embedding_matrix.shape[1],
              'enc_units': 1024,
              'dec_units': 1024,
              'attn_units': 256,
              'max_train_steps': 1250,
              'save_batch_train_data': False,
              'load_batch_train_data': False,
              'min_dec_steps': 4,
              'checkpoints_save_steps': 5}
    return params
