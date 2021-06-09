"""List the approved models versions if any"""
from dotenv import load_dotenv
from sts.utils import get_sm_session
from botocore.exceptions import ClientError
import sagemaker
import logging
import os

_l = logging.getLogger()
logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
_l.addHandler(consoleHandler)
_l.setLevel(logging.INFO)
load_dotenv()


def main():
    AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'eu-west-1')
    AWS_PROFILE = os.getenv('AWS_PROFILE', None)
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', None)
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', None)
    _, sm_client, _, _ = get_sm_session(
        region=AWS_DEFAULT_REGION,
        profile_name=AWS_PROFILE,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    MODEL_PACKAGE_GROUP_NAME = os.getenv(
        'MODEL_PACKAGE_GROUP_NAME', 'sts-sklearn-grp')

    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            _l.debug("Getting more packages for token: {}".format(
                response["NextToken"]))
            response = sm_client.list_model_packages(
                ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = (
                f"No approved ModelPackage found for ModelPackageGroup: {MODEL_PACKAGE_GROUP_NAME}"
            )
            _l.error(error_message)
            raise Exception(error_message)

        # Return the pmodel package arn
        for m in approved_packages:
            model_package_arn = m["ModelPackageArn"]
            _l.debug(f"Identified approved model package: {m}")
            print(
                f"Identified approved model package: {model_package_arn}")
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        _l.error(error_message)
        raise Exception(error_message)


if __name__ == '__main__':
    main()
