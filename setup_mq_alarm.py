"""Setup the schedule model quality monitor

Assumes the shelude is called "mq-mon-sch-sts"
"""
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor import CronExpressionGenerator
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.s3 import S3Uploader
from io import StringIO
import numpy as np
import sagemaker

from dotenv import load_dotenv
from sts.utils import get_sm_session, load_dataset
import os
import pprint
import json
import argparse
import botocore
import logging
import datetime
import time

load_dotenv()


_l = logging.getLogger()
logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
_l.addHandler(consoleHandler)
_l.setLevel(logging.INFO)
load_dotenv()


def json_default(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()


def main(resources):

    # configurarion
    AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'eu-west-1')
    AWS_PROFILE = os.getenv('AWS_PROFILE', None)
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', None)
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', None)
    b3_session, _, _, sm_session = get_sm_session(
        region=AWS_DEFAULT_REGION,
        profile_name=AWS_PROFILE,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    BASE_JOB_PREFIX = os.getenv('BASE_JOB_PREFIX', 'sts')
    ROLE_ARN = os.getenv('AWS_ROLE', sagemaker.get_execution_role())
    outputs = resources

    bucket = sm_session.default_bucket()
    prefix = "{}/{}".format(
        BASE_JOB_PREFIX,
        resources['endpoint']['name']
    )
    if 'monitor' not in resources:
        raise ValueError("Monitoring not enabled")

    if 's3_capture_upload_path' not in resources['monitor']:
        raise ValueError("Monitoring not enabled")

    if 'model-quality' not in resources['monitor']:
        raise ValueError("Setup model quality monitor first")

    monitor_schedule_name = resources['monitor']['model-quality'].get(
        'schedule_name')
    namespace = "aws/sagemaker/Endpoints/model-metrics"
    cw_client = b3_session.client('cloudwatch')
    cw_dimensions = [
        {"Name": "Endpoint", "Value": resources['endpoint']['name']},
        {"Name": "MonitoringSchedule", "Value": monitor_schedule_name},
    ]
    metric_list = []

    print("Getting metrics ...", end="", flush=True)
    while len(metric_list) == 0:
        paginator = cw_client.get_paginator('list_metrics')
        for response in paginator.paginate(
                Dimensions=cw_dimensions, Namespace=namespace):
            model_quality_metrics = response["Metrics"]
            for metric in model_quality_metrics:
                metric_list.append(metric["MetricName"])

        time.sleep(3)
        print(".", end="", flush=True)
    print(f"Metrics: {pprint.pformat(metric_list)}")

    # save outputs to a file
    with open('deploymodel_out.json', 'w') as f:
        json.dump(outputs, f, default=json_default)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploymodel-output", type=str, required=False, 
        default='deploymodel_out.json',
        help="JSON output from the deploy script"
    )

    args, _ = parser.parse_known_args()
    print(f"Using deploy info {args.deploymodel_output}")
    with open(args.deploymodel_output) as f:
        data = json.load(f)
    main(data)
