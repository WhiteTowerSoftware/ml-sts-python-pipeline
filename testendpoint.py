"""Send traffic to the endpoint

This uses the test.csv dataset"""
from sts.utils import get_sm_session
from sagemaker.deserializers import CSVDeserializer
from sagemaker.serializers import JSONSerializer
from sagemaker.sklearn.model import SKLearnPredictor
from io import StringIO
from dotenv import load_dotenv
import random
import os
import csv
import argparse
import json
import pathlib
import progressbar


load_dotenv()


def main(deploy_data, train_data):
    outputs = {'inferences': []}
    # demo dataset
    test_data_s3_uri = "s3://sts-datwit-dataset/stsmsrpc.txt"
    # --

    # AWS especific
    AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'eu-west-1')
    AWS_PROFILE = os.getenv('AWS_PROFILE', None)
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', None)
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', None)
    b3_session, sm_client, sm_runtime, sm_session = get_sm_session(
        region=AWS_DEFAULT_REGION,
        profile_name=AWS_PROFILE,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    # Load a predictor using the endpoint name
    predictor = SKLearnPredictor(
        deploy_data['endpoint']['name'],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=CSVDeserializer()
    )

    # read test data from online
    # test_data = csv.reader(
    #     StringIO(S3Downloader.read_file(test_data_s3_uri)),
    #     delimiter='\t'
    # )

    # read test data locally,to avoid downloading from intenet, uncomment
    # when necessary, comment before submit and uncomment the previus block.
    test_data = csv.reader(
        StringIO(pathlib.Path('stsmsrpc.txt').read_text()),
        delimiter='\t'
    )

    # skip first (header) row
    next(test_data)

    test_data_rows = []
    for row in test_data:
        try:
            test_data_rows.append({
                "payload": {
                    "s1": row[3],
                    "s2": row[4]
                },
                # will use the cat of the ids on stsmsrpc.txt
                "inference_id": f"{row[1]}{row[2]}",
            })
        except Exception:
            pass

    print(
        f"Sending trafic to the endpoint: {deploy_data['endpoint']['name']}")

    # do a sample of the the rows in 'stsmsrpc.txt'
    x_test_rows = random.sample(test_data_rows, 50)
    with progressbar.ProgressBar(max_value=len(x_test_rows)) as bar:
        for index, x_test_row in enumerate(x_test_rows, start=1):

            result = predictor.predict(
                x_test_row.get('payload'),
                inference_id=x_test_row.get('inference_id')
            )

            outputs['inferences'].append(
                {
                    x_test_row.get('inference_id'): {
                        'input': x_test_row.get('payload'),
                        'result': result
                    }
                }
            )

            # show progress
            bar.update(index)

    with open('testendpoint_out.json', 'w') as f:
        json.dump(outputs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploymodel-output", type=str, required=False,
        default='deploymodel_out.json',
        help="JSON output from the deploy script"
    )
    parser.add_argument(
        "--trainmodel-output", type=str, required=False,
        default='trainmodel_out.json',
        help="JSON output from the train script"
    )

    args, _ = parser.parse_known_args()
    print(f"Using deploy info {args.deploymodel_output}")
    print(f"Using training info {args.trainmodel_output}")
    with open(args.deploymodel_output) as f:
        deploy_data = json.load(f)

    with open(args.trainmodel_output) as f:
        train_data = json.load(f)

    main(deploy_data, train_data)
