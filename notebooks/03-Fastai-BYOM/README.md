# Machine Learning Workshop
This directory contains an example of bringing your own framework/library to SageMaker. It is using the OpenSource library called [fast.ai](https://github.com/fastai/fastai) which is used in the popular MOOC fast.ai [MOOC course](http://course.fast.ai/) and is based on [PyTorch](https://pytorch.org/).

## Step 1 - Create the SageMaker Notebook instance

We first need to create a Notebook instance and install the fast.ai [conda](https://conda.io/docs/user-guide/getting-started.html) environment with the library and all it's dependencies.

To do this click the **Launch Template** button below to create a new SageMaker notebook instance with a Lifecycle Configuration that installs the fast.ai library when the notebook instance is started. It also uses the *ml.p3.2xlarge* instance type with the [Nvidia Volta V100](https://www.nvidia.com/en-us/data-center/tesla-v100/) GPU to build and train the fast.ai based Deep Learning models.

[![CloudFormation](../../img/cfn-launch-stack.png)](https://eu-west-1.console.aws.amazon.com/cloudformation/home?region=eu-west-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fmlworkshop%2Ffastai-nb-instance.yml&stackName=FastaiNotebookStack&param_NotebookInstanceType=ml.p3.2xlarge)

## Step 2 - Open the Fastai example notebook

Open the Jupyter notebook web console by selecting the SageMaker notebook instance called *FastaiNotebookInstance* and clicking the option **Open** in the AWS Management Console.

Navigate to the directory **
