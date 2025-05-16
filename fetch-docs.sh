#!/bin/bash

set -e

cd /opt/groundlight/
mkdir -p docs/sdk
mkdir -p src

cd src
if [ -d python-sdk ]; then
    cd python-sdk
    git pull
    cd ..
else
    git clone --depth 1 --filter=blob:none \
    https://github.com/groundlight/python-sdk
fi
cd python-sdk/docs
cp -r docs/* /opt/groundlight/docs/sdk


