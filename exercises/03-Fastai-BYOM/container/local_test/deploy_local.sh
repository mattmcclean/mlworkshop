#!/bin/sh

image=$1
test_path=$2

if [ -z $1 ] || [ -z $2 ]; then
    echo "ERROR: Must provide a Docker image name and test_path directory"
    exit 1
fi

docker run -it -v ${test_path}:/opt/ml -p 8080:8080 --rm ${image} serve
