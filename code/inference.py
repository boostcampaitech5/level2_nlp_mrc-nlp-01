import os
import pandas as pd
import json
from preprocessing import convert_to_reader_train
from haystack.document_stores import ElasticsearchDocumentStore


def inference(config, pipeline):
    """test데이터셋을 가지고 예측을 실행합니다. 예측의 결과는 output폴더에 저장됩니다.

    Args:
        config: config.path, config.inference 사용
        pipeline: 예측에 사용할 pipeline
    """

    ## 예측 실행
    test_path = os.path.join(config.path.data_dir, config.path.test_for_inference)
    test_data = pd.read_csv(test_path)

    prediction = pipeline.run_batch(
        queries=list(test_data['question'].values),
        params={
            "Retriever": {"top_k": config.inference.retriever_top_k},
            "Reader": {"top_k": config.inference.reader_top_k}
        },
    )

    ## preds to output
    outputs = {}
    output_path = os.path.join(config.path.output_dir, 'outputs.json')
    for i in range(len(test_data)):
        outputs[test_data['id'][i]] = prediction['answers'][i][0].answer
    ## 결과 저장
    with open(output_path, 'w', encoding='UTF-8') as f:
        json.dump(outputs, f, ensure_ascii=False)
        