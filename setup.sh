#!/bin/sh

if [ ! -d ./.venv ]; then \
  echo "Creating Venv..."; \
  python3 -m venv .venv; \
  .venv/bin/pip install -r ./requirement.txt ; \
  echo "Venv created." ; \
fi

if [ ! -d ./datasets -o ! -f ./datasets/dataset_test.csv -o ! -f ./dataset_train.csv ]; then \
  echo "Importing Datasets..."; \
  wget https://cdn.intra.42.fr/document/document/36200/datasets.tgz; \
  tar -xf ./datasets.tgz; \
  rm -rf ./datasets.tgz; \
  echo "Datasets imported !"; \
fi

. ./.venv/bin/activate
echo "Venv sourced."