import pandas as pd
import json

def convert_to_reader_train(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    data = []
    for i in range(len(df)):
        title = df['title'][i]
        paragraphs = [{'qas': [{'question': df['question'][i], 'id': df['id'][i], 'answers': [{'text': eval(df['answers'][i])['text'][0], 'answer_start': eval(df['answers'][i])['answer_start'][0]}], 'is_impossible': False}], 'context': df['context'][i]}]
        data.append({'title': title, 'paragraphs': paragraphs})
    data = {'data': data}
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    return output_path