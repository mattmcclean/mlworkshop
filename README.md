# Machine Learning Workshop
This repo holds the Jupyter notebooks and other resources such as CFN templates to run the SageMaker Machine Learning workshop led by Matt McClean from AWS.

## Setup

Follow the instructions below to setup your S3 bucket, IAM role and SageMaker Notebook Instance.

We will use [CloudFormation](https://aws.amazon.com/cloudformation/) to create our resources via a template file. To do this,

1. Click the **Launch Template** button below to open the AWS CloudFormation Web Console to create a new CloudFormation stack. Click through the options and select the SageMaker instance type. The **ml.t2.medium** option is part of the AWS Free Tier. See the SageMaker [pricing page](https://aws.amazon.com/sagemaker/pricing/) for more details.

[![CloudFormation](img/cfn-launch-stack.png)](https://eu-west-1.console.aws.amazon.com/cloudformation/home?region=eu-west-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fmlworkshop%2Fcfn.yml&stackName=MLWorkshopStack&param_NotebookInstanceType=ml.t2.medium)

Take note of the resources created including:
 - **S3 bucket** where the training data and models will be stored
 - **IAM service role** allowing SageMaker access various AWS services
 - **SageMaker Notebook Instance** to run the exercises in the workshop.

![Screenshot](img/cfn-outputs.png)

2. Open the SageMaker Management console and select the SageMaker notebook instance named: **AmsterdamWorkshopInstance** as per the screenshot below.

![Screenshot](img/sagemaker-nb.png)


