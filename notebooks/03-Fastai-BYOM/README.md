# fast.ai BYOM exercise
This directory contains an example of bringing your own framework/library to SageMaker. It is using the OpenSource library called [fast.ai](https://github.com/fastai/fastai) which is used in the popular MOOC fast.ai [MOOC course](http://course.fast.ai/) and is based on [PyTorch](https://pytorch.org/).

## Step 1 - Create the SageMaker Notebook instance

We first need to create a Notebook instance and install the fast.ai [conda](https://conda.io/docs/user-guide/getting-started.html) environment with the library and all it's dependencies.

To do this click the **Launch Template** button below to create a new SageMaker notebook instance with a Lifecycle Configuration that installs the fast.ai library when the notebook instance is started. It also uses the *ml.p3.2xlarge* instance type with the [Nvidia Volta V100](https://www.nvidia.com/en-us/data-center/tesla-v100/) GPU to build and train the fast.ai based Deep Learning models.

[![CloudFormation](../../img/cfn-launch-stack.png)](https://eu-west-1.console.aws.amazon.com/cloudformation/home?region=eu-west-1#/stacks/create/review?filter=active&templateURL=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fmmcclean-public-files%2Fmlworkshop%2Ffastai-nb-instance.yml&stackName=FastaiNotebookStack&param_NotebookInstanceType=ml.p3.2xlarge)

## Step 2 - Open the Fastai example notebook

Open the Jupyter notebook web console by selecting the SageMaker notebook instance called *FastaiNotebookInstance* and clicking the option **Open** in the AWS Management Console.

Navigate to the directory *mlworkshop/notebooks/03-Fastai-BYOM* and open the notebook named **03-fastai_caltech256_train.ipynb**. Run through the steps of the notebook to train and deploy your fast.ai based model.

Your model will then be saved to your S3 bucket (replacing *<account_id>* with your AWS account id and *<region>* with the region name):

```
s3://sagemaker-<account_id>-<region>/models/caltech256_fastai/model.tar.gz
```

## Step 3 - Delete/Stop your fast.ai notebook instance

Now that you have trained your fast.ai model you no longer need it so either stop it via the AWS Management Console or delete the CloudFormation stack name: **FastaiNotebookStack**.

## Step 4 - Deploy your fast.ai model to SageMaker hosting endpoint

Open your original SageMaker notebook instance called **MLWorkshopInstance** and open the Jupyter notebook in the folder *mlworkshop/notebooks/03-Fastai-BYOM* titled **03-fastai_caltech256_predict.ipynb**. Run through the steps of the notebook to create your SageMaker model and deploy the endpoint.

The final steps show you how to call the SageMaker endpoint and check the results.
