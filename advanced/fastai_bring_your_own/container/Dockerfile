# Build an image that can do training and inference in SageMaker
# This is an image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM python:3.6.5-slim-stretch

MAINTAINER Amazon AI <mmcclean@amazon.com>


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         nginx \
         ca-certificates \
         libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*


# Here we get all python packages.
RUN pip install flask gevent gunicorn future
RUN pip install boto3 pyyaml dill numpy opencv-python-headless \
    http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl \
    https://s3-eu-west-1.amazonaws.com/mmcclean-public-files/fastai-lib.zip && \ 
    rm -rf /root/.cache

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY fastai_predict /opt/program
WORKDIR /opt/program

