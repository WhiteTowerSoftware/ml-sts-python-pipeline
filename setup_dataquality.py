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


def main(resources, train_data):

    # configurarion
    AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'eu-west-1')
    AWS_PROFILE = os.getenv('AWS_PROFILE', None)
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', None)
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', None)
    _, _, _, sm_session = get_sm_session(
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
    
    baseline_prefix = prefix + "/data-quality/baselining"
    baseline_data_prefix = baseline_prefix + "/data"
    baseline_results_prefix = baseline_prefix + "/results"
    dq_schedule_output_prefix = baseline_prefix + "/monitoring"
    baseline_data_uri = "s3://{}/{}/baseline.csv".format(
        bucket, baseline_data_prefix)
    baseline_results_uri = "s3://{}/{}".format(bucket, baseline_results_prefix)
    dq_mon_schedule_output_s3_path = "s3://{}/{}".format(
        bucket, dq_schedule_output_prefix)
    outputs['monitor'].update({
        'data-quality': {
            'baseline_data': baseline_data_uri,
            'baseline_results': baseline_results_uri,
            'output': dq_mon_schedule_output_s3_path
        }
    })
    _l.info("Data Quality Baseline data uri: {}".format(baseline_data_uri))
    _l.info("Data Quality Baseline results uri: {}".format(
        baseline_results_uri))

    # data prep, for data quality we need to use the train data set
    # but is necessary to drop the label column (Y_train)
    _l.info(f"Loadding {train_data['train']['train']}")
    train_set = load_dataset(
        train_data['train']['train'], 'train.csv', 
        sagemaker_session=sm_session)
    # drop Y_train
    train_set.drop(train_set.columns[0], axis=1, inplace=True)
    train_set_file = StringIO()
    # save basesile dataset
    train_set.to_csv(train_set_file, header=False, index=False)
    S3Uploader.upload_string_as_file_body(
        train_set_file.getvalue(), 
        desired_s3_uri=baseline_data_uri,
        sagemaker_session=sm_session
    )
    # baseline_data_uri
    # --

    # Create a baselining job with training dataset
    _l.info("Executing a baselining job with training dataset")
    _l.info(f"baseline_data_uri: {baseline_data_uri}")
    my_monitor = DefaultModelMonitor(
        role=ROLE_ARN,
        sagemaker_session=sm_session,
        max_runtime_in_seconds=1800
    )

    # execute the baseline job for data quality
    my_monitor.suggest_baseline(
        baseline_dataset=baseline_data_uri,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=baseline_results_uri,
        wait=True, logs=False,
    )
    baseline_job = my_monitor.latest_baselining_job

    # after the jobs run this will output the suggested contrains and 
    # statistics, both are saved to a s3 bucket, see deploymodel_out.json
    # for monitor.data-quality-baseline.results_uri
    contrains = baseline_job.suggested_constraints()
    stats = baseline_job.baseline_statistics()
    _l.debug("suggested baseline contrains")
    _l.debug(
        pprint.pformat(contrains.body_dict["features"]))
    _l.debug("suggested baseline statistics")
    _l.debug(
        pprint.pformat(contrains.body_dict["features"]))
    _l.debug("Data Quality monitoring config:")
    _l.debug(
        pprint.pformat(contrains.body_dict["monitoring_config"]))
    # add monitoring config to outputs
    outputs['monitor'].update({
        'data-quality': {
            'contrains': contrains.body_dict["features"],
            'statistics': stats.body_dict['features'],
            'config': contrains.body_dict['monitoring_config']
        }
    })

    print("")
    print("For details on Data Quality violations for various problem types, refer to")
    print("https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-interpreting-violations.html")

    monitor_schedule_name = (
        f"{BASE_JOB_PREFIX}-dq-sch-{datetime.datetime.utcnow():%Y-%m-%d-%H%M}"
    )
    _l.info(f"Monitoring schedule name: {monitor_schedule_name}")
    outputs['monitor']['data-quality'].update({
        'schedule_name': monitor_schedule_name
    })

    # schedule a job to run hourly over the data capture in the model endpoint
    # to search for problems related to data quality, this will produce
    # metrics on cloudwatch. You can see charts and the result of the 
    # job executions trought SageMaker Studio.
    my_monitor.create_monitoring_schedule(
        monitor_schedule_name=monitor_schedule_name,
        endpoint_input=resources['endpoint']['name'],
        output_s3_uri=dq_mon_schedule_output_s3_path,
        statistics=my_monitor.baseline_statistics(),
        constraints=my_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True
    )

    dq_schedule_details = my_monitor.describe_schedule()
    while dq_schedule_details['MonitoringScheduleStatus'] == 'Pending':
        _l.info(f'Waiting for {monitor_schedule_name}')
        time.sleep(3)
        dq_schedule_details = my_monitor.describe_schedule()
    _l.debug(
        f"Model Quality Monitor - schedule details: {pprint.pformat(dq_schedule_details)}")
    _l.info(
        f"Model Quality Monitor - schedule status: {dq_schedule_details['MonitoringScheduleStatus']}")

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
    parser.add_argument(
        "--trainmodel-output", type=str, required=False, 
        default='trainmodel_out.json',
        help="JSON output from the train script"
    )

    args, _ = parser.parse_known_args()
    print(f"Using deploy info {args.deploymodel_output}")
    with open(args.deploymodel_output) as f:
        data = json.load(f)
    _l.info(f"Using training info {args.trainmodel_output}")
    with open(args.trainmodel_output) as f:
        train_data = json.load(f)
    main(data, train_data)
