import pandas as pd
from datasets import load_from_disk, Dataset

'''
함수 curri_extract를 train.py 파일에서 불러온 후 compute_metrics 내에 
curri_extract(p.predictions, p.label_ids) 한 줄 추가하며 curriculum_pre_roberta.csv 파일 생성

위 코드는 train data를 기반으로 evalutation 을 통해 어려움을 측정하는 방식이므로
overfitting의 경우 거의 100%에 준하는 EM이 측정되므로 1 epoch 정도로 학습 권장
'''

def curri_extract(a, b):
    '''
    Input
    a : predictions
    b : label_ids
    
    Output
    train_dataset에 대한 예측 / 정답 data
    '''
    pred_id = []
    pred_label = []
    target_id = []
    target_label = []
    
    for i in range(len(a)):
        pred_id.append(a[i]['id'])
        pred_label.append(a[i]['prediction_text'])
        target_id.append(b[i]['id'])
        target_label.append(b[i]['answers']['text'][0])

    my_dict = {'pred_id' : pred_id, 'pred_label':pred_label, 'target_id':target_id, 'target_label':target_label}
    df = pd.DataFrame(my_dict)
    df.to_csv('curriculum_pre_roberta', index = False)
    



if __name__ == "__main__": 
    df = pd.read_csv('curriculum_pre_roberta.csv')

    df['EM'] = (df['pred_label'] == df['target_label']).astype(int)

    df_easy = df[df['EM'] == 1]
    df_hard = df[df['EM'] == 0]

    easy_list = list(df_easy['pred_id'])
    hard_list = list(df_hard['pred_id'])


    # Arrow 파일을 Dataset으로 불러오기
    dataset = load_from_disk('../data/train_dataset/train')

    def easy_check(df):
        return df['id'] in easy_list

    def hard_check(df):
        return df['id'] in hard_list

    # 필터링 적용
    easy_dataset = dataset.filter(easy_check)
    hard_dataset = dataset.filter(hard_check)

    print(f'easy data set size : {easy_dataset.num_rows}')
    print(f'hard data set size : {hard_dataset.num_rows}')

    # Dataset을 Arrow 파일로 저장
    easy_dataset.save_to_disk('easy_train')
    hard_dataset.save_to_disk('hard_train')
    
    print(f'finished')

