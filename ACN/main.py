import argparse

from trainer import Trainer, OPTIMIZER_LIST
from utils import init_logger, build_vocab, download_vgg_features, set_seed
from data_loader import load_data


def main(args):
    init_logger()
    set_seed(args)
    download_vgg_features(args)
    build_vocab(args)

    train_dataset = load_data(args, mode="train")
    dev_dataset = load_data(args, mode="dev")
    test_dataset = load_data(args, mode="test")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")

    if args.do_finetune:
        trainer.load_model_finetune()
        trainer.finetune()
        trainer.load_finetune_model()
        trainer.evaluate("test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path for saving model")
    parser.add_argument("--finetune_model_dir", default="./finetune_model", type=str, help="Path for saving model")
    parser.add_argument("--wordvec_dir", default="./wordvec", type=str, help="Path for pretrained word vector")
    parser.add_argument("--vocab_dir", default="./vocab", type=str)

    parser.add_argument("--train_file", default="weibo-zh-train.txt", type=str, help="Train file")
    parser.add_argument("--dev_file", default="weibo-zh-valid.txt", type=str, help="Dev file")
    parser.add_argument("--test_file", default="weibo-zh-test.txt", type=str, help="Test file")
    # parser.add_argument("--w2v_file", default="/home/jiyuanze/MUSE/dumped/debug/zh->en/vectors-motley.txt", type=str, help="Pretrained word vector file")
    # parser.add_argument("--w2v_file", default="/home/jiyuanze/MUSE/dumped/debug/en->zh/vectors-zh_new.txt",
    #                     type=str, help="Pretrained word vector file")
    # parser.add_argument("--w2v_file", default="word_vector_200d.vec", type=str, help="Pretrained word vector file")
    parser.add_argument("--w2v_file", default="wiki.zh.vec", type=str, help="Pretrained word vector file")  # 300d
    parser.add_argument("--img_feature_file", default="img_vgg_features_motley.pt", type=str,help="Filename for preprocessed image features")

    parser.add_argument("--max_seq_len", default=250, type=int, help="Max sentence length")  # en:35 zh:250
    parser.add_argument("--max_word_len", default=30, type=int, help="Max word length")

    parser.add_argument("--word_vocab_size", default=23204, type=int, help="Maximum size of word vocabulary")
    parser.add_argument("--char_vocab_size", default=3978, type=int, help="Maximum size of character vocabulary")

    parser.add_argument("--word_emb_dim", default=300, type=int, help="Word embedding size")    # en:200 zh:300
    parser.add_argument("--char_emb_dim", default=30, type=int, help="Character embedding size")
    parser.add_argument("--final_char_dim", default=50, type=int, help="Dimension of character cnn output")
    parser.add_argument("--hidden_dim", default=200, type=int, help="Dimension of BiLSTM output, att layer (denoted as k) etc.")

    parser.add_argument("--kernel_lst", default="2,3,4", type=str, help="kernel size for character cnn")
    parser.add_argument("--num_filters", default=32, type=int, help=" Number of filters for character cnn")

    parser.add_argument('--seed', type=int, default=444, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation")
    parser.add_argument("--optimizer", default="adam", type=str, help="Optimizer selected in the list: " + ", ".join(OPTIMIZER_LIST.keys()))
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate")  # lr:0.001  finetune:0.0001
    parser.add_argument("--num_train_epochs", default=24.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--slot_pad_label", default="[pad]", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")
    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--logging_steps', type=int, default=188, help="Log every X updates steps.")  # twi:250 weibo:188
    parser.add_argument('--save_steps', type=int, default=188, help="Save checkpoint every X updates steps.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_finetune", action="store_true", help="Whether to run fine-tuning.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--no_w2v", action="store_true", help="Not loading pretrained word vector")

    args = parser.parse_args()

    # For VGG16 img features (DO NOT change this part)
    args.num_img_region = 49
    args.img_feat_dim = 512
    # For VinVL img features
    # args.num_img_region = 10
    # args.img_feat_dim = 2048

    main(args)
