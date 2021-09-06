from trainer import PICOSentClassTrainer
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train pico sentence classifiation using pre-trained BLUE model.')
    parser.add_argument('--corpus_path', default='../data/ebm_pico_p/', help='the path to the corpus folder.')
    parser.add_argument('--output_path', default='../exps/pubmed/ebm_pico_p/', help='the path to the ouput folder.')
    parser.add_argument('--train_file', default='train.json', help='the name of the train file.')
    parser.add_argument('--dev_file', default='dev.json', help='the name of the dev file.')
    parser.add_argument('--test_file', default='test.json', help='the name of the test file.')

    parser.add_argument('--label_name', default='aggregation', help='the sentence label generation type',
                        choices=['major', 'minor', 'aggregation'])
    parser.add_argument('--bert_name', default='pubmed', help='the pre-trained bert model name',
                        choices=['pubmed', 'pubmed_mimic'])
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='batch size per GPU for evaluation; bigger batch size makes training faster')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='batch size per GPU for training')
    parser.add_argument('--max_len', type=int, default=512,
                        help='length that documents are padded/truncated to')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='gradient accumulation steps during training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='sentence classification training epochs; 3-5 usually is good depending on dataset size (smaller dataset needs more epochs)')
    parser.add_argument('--eval_metric', default='f1', help='the evaluation metric',
                        choices=['acc', 'f1'])
    args = parser.parse_args()
    print(args)
    trainer = PICOSentClassTrainer(args)
    # Training Sentence Classification Task based on different ways of label generation
    trainer.sentence_classification(epochs=args.epochs)


if __name__ == "__main__":
    main()
