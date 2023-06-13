import pandas as pd
from datasets import load_from_disk
import json

if __name__ == 'main':
    """train.csv, valid.csv, test.csv 생성"""
    data = load_from_disk('../data/train_dataset')
    test = load_from_disk('../data/test_dataset')

    train = data['train']
    valid = data['validation']

    train = pd.DataFrame(train)
    valid = pd.DataFrame(valid)
    test = pd.DataFrame(test['validation'])

    train.to_csv('../data/train.csv', index=False)
    valid.to_csv('../data/valid.csv', index=False)
    test.to_csv('../data/test.csv', index=False)