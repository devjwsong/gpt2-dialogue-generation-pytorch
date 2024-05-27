# gpt2-dialogue-generation-pytorch

This is a multi-turn chatbot project using the pre-trained GPT-2[[1]](#1) introduced in *How to build a State-of-the-Art Conversational AI with Transfer Learning*[[2]](#2).

Especially, this repository uses the GPT-2 Language Modeling Head model which has one additional linear layer to conduct Language Modeling task to consider the dialogue contexts and make a proper next response.

I did not include the persona information unlike the original version.

<br/>

---

### Arguments

**Arguments for data loading**

| Argument       | Type    | Description                                                  | Default   |
| -------------- | ------- | ------------------------------------------------------------ | --------- |
| `data_dir`     | `str`   | The name of the parent directory where data files are stored. | `"data"`  |
| `train_prefix` | `str`   | The prefix of the train data files' name.                    | `"train"` |
| `valid_prefix` | `str`   | The prefix of the validation data files' name.               | `"valid"` |
| `train_frac`   | `float` | The ratio of the conversations to be included in the train set. | `0.85`    |
| `model_type`   | `str`   | The model type of GPT-2. (`"gpt2"`, `"gpt2-medium"`, `"gpt2-large"`, or `"gpt2-xl"`) | `"gpt2"`  |

<br/>

**Arguments for training**

| Argument              | Type              | Description                                                  | Default               |
| --------------------- | ----------------- | ------------------------------------------------------------ | --------------------- |
| `seed` | `int` | The random seed. | `0` |
| `data_dir`            | `str`        | The name of the parent directory where data files are stored. | `"data"`              |
| `train_prefix` | `str`       | The prefix of the train data files' name.                    | `"train"`             |
| `valid_prefix` | `str`       | The prefix of the validation data files' name.               | `"valid"`    |
| `model_type` | `str` | The model type of GPT-2. (`"gpt2"`, `"gpt2-medium"`, `"gpt2-large"`, or `"gpt2-xl"`) | `"gpt2"` |
| `bos_token`          | `str`        | The BOS token.           | `"<bos>"`             |
| `sp1_token`  | `str`       | The speaker1 token.                | `"<sp1>"`      |
| `sp2_token`  | `str`       | The speaker2 token.              | `"<sp2>"`     |
| `gpu`           | `str`        | The index of GPU to use. | `"0"`              |
| `lr`      | `float` | The learning rate.                                       | `2e-5`            |
| `warmup_ratio` | `float` | The ratio of warmup steps to the total training steps. | `0.1` |
| `batch_size`          | `int` | The batch size.                                              | `8`                  |
| `num_workers` | `int` | The number of workers for data loading. | `0` |
| `num_epochs`          | `int` | The number of total epochs.   | `10`                  |
| `max_len`      | `int`   | The maximum length of input sequence.                        | `1024`                |
| `max_turns`   | `int`   | The maximum number of dialogue histories to include. | `5`                   |
| `ckpt_dir`            | `str`        | The path for saved checkpoints.                              | `"saved_models"`      |
| `ckpt_name`            | `str`        | The default name for the trained model. (without extension)                              | *YOU MIGHT SPECIFY*  |

<br/>

**Arguments for inference**

| Argument      | Type    | Description                                            | Default              |
| ------------- | ------- | ------------------------------------------------------ | -------------------- |
| `seed`        | `int`   | The random seed.                                       | `0`                  |
| `model_path`  | `str`   | The path to the model in HuggingFace Hub.              | *YOU SHOULD SPECIFY* |
| `gpu`         | `str`   | The index of GPU to use.                               | `"0"`                |
| `max_turns`   | `int`   | The maximum number of dialogue histories to include.   | `5`                  |
| `top_p`       | `float` | The top-p value for nucleus sampling decoding.         | `0.8`                |
| `end_command` | `str`   | The command to stop the conversation when inferencing. | `"Abort!"`           |

<br/>

---

### Datasets

By default, I propose the codes for downloading the datasets and preprocessing.

There are 4 types of the default datasets as follows.

<br/>

- DailyDialog[[3]](#3)
- EmpatheticDialogues[[4]](#4)
- Persona-Chat[[5]](#5)
- BlendedSkillTalk[[6]](#6)

<br/>

---

### How to run

#### Fine-tuning a new model

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Download & Preprocess all datasets.

   ```shell
   sh exec_load_data.sh
   ```

   After running it, you will have the following data directory structure if you follow the default argument setting.

   ```
   data
   └--gpt2
       └--train_utters.pickle
       └--train_ids.pickle
       └--valid_utters.pickle
       └--valid_ids.pickle
   ```

   <br/>

3. Run the following command to train the model.

   If you want to train it starting from a specific checkpoint, add the argument `ckpt_name` and make sure to notify the proper checkpoint name.

   ```shell
   sh exec_train.sh
   ```
   

<br/>

#### Loading a fine-tuned model & Chatting with it

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. If you already have a model which has been pushed to HuggingFace's Hub, you can load it and chat with the model. If you don't have a model, you can use the model I fine-tuned: https://huggingface.co/devjwsong/gpt2-open-domain-dialogue. To do that, change the argument `--model_path` in `exec_infer.sh` into a corresponding path.

   After that, run below command.

   ```shell
   sh exec_infer.sh
   ```

<br/>

---

### References

<a id="1">[1]</a> Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.([http://www.persagen.com/files/misc/radford2019language.pdf](http://www.persagen.com/files/misc/radford2019language.pdf))

<a id="2">[2]</a> How to build a State-of-the-Art Conversational AI with Transfer Learning . (2019, May 9). ([https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313))

<a id="3">[3]</a> Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). Dailydialog: A manually labelled multi-turn dialogue dataset. *arXiv preprint arXiv:1710.03957*. ([https://arxiv.org/abs/1710.03957](https://arxiv.org/abs/1710.03957))

<a id="4">[4]</a> Rashkin, H., Smith, E. M., Li, M., & Boureau, Y. L. (2018). Towards empathetic open-domain conversation models: A new benchmark and dataset. *arXiv preprint arXiv:1811.00207*. ([https://arxiv.org/abs/1811.00207](https://arxiv.org/abs/1811.00207))

<a id="5">[5]</a> Zhang, S., Dinan, E., Urbanek, J., Szlam, A., Kiela, D., & Weston, J. (2018). Personalizing dialogue agents: I have a dog, do you have pets too?. *arXiv preprint arXiv:1801.07243*. ([https://arxiv.org/abs/1801.07243](https://arxiv.org/abs/1801.07243))

<a id="6">[6]</a> Smith, E. M., Williamson, M., Shuster, K., Weston, J., & Boureau, Y. L. (2020). Can You Put it All Together: Evaluating Conversational Agents' Ability to Blend Skills. *arXiv preprint arXiv:2004.08449*. ([https://arxiv.org/abs/2004.08449](https://arxiv.org/abs/2004.08449))
