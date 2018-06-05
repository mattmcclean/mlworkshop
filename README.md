# Machine Learning Workshop
This repo holds the Jupyter notebooks and other resources such as CFN templates to run the SageMaker Machine Learning workshop led by Matt McClean from AWS.

## Setup

Follow the instructions below to setup your S3 bucket, IAM role and SageMaker Notebook Instance.

We will use [CloudFormation](https://aws.amazon.com/cloudformation/) to create our resources via a template file. To do this, 

[![CloudFormation](cfn-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=eu-west-1#/stacks/new?stackName=MLWorkshopStack&templateURL=https://raw.githubusercontent.com/mattmcclean/mlworkshop/master/cfn.yml)
