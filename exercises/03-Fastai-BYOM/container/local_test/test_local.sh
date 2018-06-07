#!/bin/bash

file="/tmp/test.jpg"

if [ -z "$1" ]; then
    url="http://www.vision.caltech.edu/Image_Datasets/Caltech256/images/008.bathtub/008_0007.jpg"
else
    url=$1
fi

wget -O ${file} ${url}

echo "Using image file: ${file} from URL: ${url}"

response=$(curl -s  -X POST --data-binary @${file} -H "Accept: application/json" -H "Content-Type: image/jpeg" http://localhost:8080/invocations)
echo "Response is:"
command -v jq >/dev/null 2>&1 && echo $response | jq . || echo $response

