import os
import sys
import json
import pickle
import pathlib
import logging
import argparse
import evaluate
import subprocess
import pandas as pd
from utils import pd_readfile, write_jsonl
from typing import Union
from resources.factcc.modeling.model import BertPointer
from pytorch_transformers import BertConfig, BertTokenizer


class Marker():
    def __init__(self, args):
        self.df = pd.DataFrame()
        self.df['summary'] = pd_readfile(args.summary)
        if args.source is not None:
            self.df['source'] = pd_readfile(args.source)
        if args.reference is not None:
            self.df['reference'] = pd_readfile(args.reference)

    def factCC(self):
        factcc = self.df.loc[:, ['source', 'summary']]
        factcc.rename(columns={'source': 'text', 'summary': 'claim'}, inplace=True)
        factcc.insert(0, 'label', 'INCORRECT')
        print("************* creat data file for factcc *************")
        write_jsonl(factcc, "tmp/data-dev.jsonl")

        code_dir = pathlib.Path('resources/factcc/modeling/')
        data_dir = pathlib.Path('tmp/')
        ckpt_dir = pathlib.Path('resources/factcc-checkpoint/')
        fact_score_dir = pathlib.Path('tmp/fact_score.pkl')

        task_name = 'factcc_annotated'
        model_name = 'bert-base-uncased'

        subprocess.run(['python3', str(code_dir/'run.py'),
                        '--task_name', task_name,
                        '--do_eval',
                        '--eval_all_checkpoints',
                        '--do_lower_case',
                        '--overwrite_cache',
                        '--max_seq_length', '512',
                        '--per_gpu_train_batch_size', '12',
                        '--model_type', 'bert',
                        '--model_name_or_path', model_name,
                        '--data_dir', data_dir,
                        '--output_dir', ckpt_dir,
                        '--fact_score_dir', fact_score_dir])

        with open(fact_score_dir, 'rb') as fp:
            fact_score = pickle.load(fp)
        return fact_score

    def bertscore(self):
        pass

    def bartscore(self):
        pass

    def rouge(self):
        scorer = evaluate.load("rouge")
        pred = self.df['summary'].values
        ref = self.df['reference'].values
        results = scorer.compute(predictions=pred, references=ref)
        return results

    def mark(self, args):
        scores = {}
        if args.all or args.factcc:
            scores["factcc"] = self.factCC()
        if args.all or args.bertscore:
            scores["bertscore"] = self.bertscore()
        if args.all or args.bartscore:
            scores["bartscore"] = self.bartscore()
        if args.all or args.rouge:
            scores["rouge"] = self.rouge()

        with open(args.output, "w") as fp:
            json.dump(scores, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ------------------------file path--------------------------
    parser.add_argument("--source",
                        type=str,
                        required=False,
                        default=None,
                        help="The path of the source document")
    parser.add_argument("--reference",
                        type=str,
                        required=False,
                        default=None,
                        help="The path of the reference summary")
    parser.add_argument("--summary",
                        type=str,
                        required=True,
                        help="The path of the generated summary")
    parser.add_argument("--output",
                        type=str,
                        default="./results.json",
                        help="The path of the output results")

    # ------------------------metrics--------------------------
    parser.add_argument("--all",
                        action="store_true",
                        default=False,
                        help="Evaluate the generated summary under all metrics")
    parser.add_argument("--rouge",
                        action="store_true",
                        default=False,
                        help="ROUGE score")
    parser.add_argument("--bartscore",
                        action="store_true",
                        default=False,
                        help="BARTScore")
    parser.add_argument("--bertscore",
                        action="store_true",
                        default=False,
                        help="BERTScore")
    parser.add_argument("--factcc",
                        action="store_true",
                        default=False,
                        help="FactCC")

    args = parser.parse_args()
    marker = Marker(args)
    marker.mark(args)
