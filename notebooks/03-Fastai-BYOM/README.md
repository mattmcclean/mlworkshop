# Machine Learning Workshop
This directory contains an example of bringing your own framework/library to SageMaker. It is using the OpenSource library called [fast.ai](https://github.com/fastai/fastai) which is a popular


## Step 1 - Create the SageMaker Notebook instance

We first need to create a Notebook instance and install the fast.ai conda environment with the library and all it's dependencies.

To do this click the **Launch Template** button below to create a new SageMaker notebook instance with a Lifycycle Configuration that installs the fast.ai library when the notebook instance is started. It also uses a GPU based instance type (e.g. ml.p2.xlarge or ml.p3.2xlarge) to build and train the fast.ai based ML models.

[![CloudFormation](../../img/cfn-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=eu-west-1#/stacks/new?stackName=FastaiNotebookStack&templateURL=https://s3-eu-west-1.amazonaws.com/mmcclean-public-files/mlworkshop/fastai-nb-instance.yml)

## Step 2 - Open the
