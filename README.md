# witt-gpt
Scripts to download, train and inference GPT-NeoX on a grid computing platform (or your own computer if you have the computing capacity) using the Hugging Face Python library.

## How to use:

Intended use is on Unix systems (hasn't been tested on Windows).

1. Clone this repository with and change working directory:

`git clone https://github.com/PiffTheSenpai/witt-gpt `

`cd witt-gpt`

2. Download Hugging Face's Transformers library + other dependencies if you don't have them:

`pip install transformers huggingface_hub`

 (optional) `pip install torch torchvision`

3. Run `download.py` to download the model:

`python download.py`

4. Specify your questions into `prompt.txt` file and run `inference.py` to generate your responses:


## Important links with settings explanations

In order to change settings, please change respective python script, ie: `inference.py` and `train.py`.

https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox

https://huggingface.co/docs/transformers/main_classes/tokenizer

https://huggingface.co/docs/transformers/main_classes/text_generation

https://huggingface.co/docs/transformers/main_classes/trainer

https://huggingface.co/docs/transformers/internal/generation_utils

