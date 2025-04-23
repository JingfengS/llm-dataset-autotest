# Simplify your llm testing experience

## Idea

Almost for every dataset, you have to write your own script to test it. This project aims to simplify that process by providing a unified framework for testing various datasets with minimal configuration.

## Usage

#### Your own dataset

You can create your own test suite for your own dataset by taking MME_TestModel as an example. 

> There will be more information later

#### For running scripts

You can take qwen2, qwen2_5 as an example to write your own script.

Note that before running the script, you need to set the model used for testing, you can either use your own model or use chatgpt api.

``` bash
deepeval set-local-model --model-name='qwen2_5' \
--base-url='http://localhost:2000/v1/' \
--api-key="Anything because local model doesn't require an api key"
```

## Datasets

- [MME Dataset](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
- [MMMU Dataset](https://github.com/MMMU-Benchmark/MMMU)
- [MathVista](https://github.com/lupantech/MathVista/blob/main/README.md#-evaluations-on-mathvista)
- [MM-VET](https://github.com/yuweihao/MM-Vet/tree/main)
- [coco2014](https://hf-mirror.com/datasets/triciahu/coco2014val/tree/main)


### TODO:
- [ ] Add more datasets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
