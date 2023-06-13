import pandas as pd
import json

def convert_to_reader_train(csv_path, output_path):
    """CSV 파일을 squard dataset형식으로 변경하고 json파일로 저장합니다

    Args:
        csv_path: squard dataset 형식으로 변환할 csv파일 path
        output_path: 변환된 데이터를 저장할 path (반드시 json확장자)

    """
    
    df = pd.read_csv(csv_path)
    
    data = []
    for i in range(len(df)):
        title = df['title'][i]
        paragraphs = [{'qas': [{'question': df['question'][i], 'id': df['id'][i], 'answers': [{'text': eval(df['answers'][i])['text'][0], 'answer_start': eval(df['answers'][i])['answer_start'][0]}], 'is_impossible': False}], 'context': df['context'][i]}]
        data.append({'title': title, 'paragraphs': paragraphs})
    data = {'data': data}
    
    with open(output_path, 'w') as f:
        json.dump(data, f)