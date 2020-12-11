# gpt2-chatbot-pytorch

This is a multi-turn chatbot project using the pre-trained GPT-2 introduced in *How to build a State-of-the-Art Conversational AI with Transfer Learning*[[1]](#1).

Especially, this repository uses the GPT-2 LM Head model which has one additional linear layer to conduct Language Modeling task to consider the dialogue contexts and make a proper response.

<br/>

---

### Configurations

You can set various arguments by modifying `config.json` in the top directory.

The description of each variable is as follows. (Those not introduced in below table are set automatically and should not be changed.)

| Argument              | Type              | Description                                                  | Default               |
| --------------------- | ----------------- | ------------------------------------------------------------ | --------------------- |
| `data_dir`            | `String`          | The name of the parent directory where data files are stored. | `"data"`              |
| `train_name`          | `String`          | The prefix of the train data files' name.                    | `"train"`             |
| `valid_name`          | `String`          | The prefix of the validation data files' name.               | `"validation"`        |
| `train_frac`          | `Number`(`float`) | The ratio of the conversations to be included in the train set. | `0.85`                |
| `pad`                 | `String`          | The padding token.                                           | `"<pad>"`             |
| `bos`                 | `String`          | The BOS(Beginning Of Sentence) token.                        | `"<bos>"`             |
| `eos`                 | `String`          | The EOS(End Of Sentence) token.                              | `"<eos>"`             |
| `speaker1`            | `String`          | The first speaker's token.                                   | `"<speaker1>"`        |
| `speaker2`            | `String`          | The second speaker's token.                                  | `"<speaker2>"`        |
| `dialogue_split_line` | `String`          | The line for splitting each dialogue in the preprocessed data files. | `"[END OF DIALOGUE]"` |
| `device`              | `String`          | The device type. (`"cuda"` or `"cpu"`) If this is set to `"cuda"`, then the device configuration is set to `torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')`. If this variable is `"cpu"`, then the setting becomes just `torch.devcie('cpu')`. | `"cuda"`              |
| `learning_rate`       | `Number`(`float`) | The learning rate.                                           | `5e-4`                |
| `batch_size`          | `Number`(`int`)   | The batch size.                                              | `4`                   |
| `num_epochs`          | `Number`(`int`)   | The total number of iterations.                              | `10`                  |
| `max_len`             | `Number`(`int`)   | The maximum length of a sentence. (Note that this should be less than the maximum length the GPT-2 can take.) | `1024`                |
| `max_time`            | `Number`(`int`)   | The maximum length of the dialogue history to be attended.   | `5`                   |
| `nucleus_p`           | `Number`(`float`) | The ratio of the probability mass for top-$p$ sampling(nucleus sampling). | `0.9`                 |
| `ckpt_dir`            | `String`          | The path for saved checkpoints.                              | `"saved_models"`      |
| `ckpt_name`            | `String`          | The default name for the trained model. (without extension)                              | `"best_ckpt"`      |
| `end_command`         | `String`          | The command to stop the conversation when inferencing.       | `"Abort!"`            |

<br/>

---

### Datasets

By default, I propose the codes for downloading the datasets and preprocessing.

There are 4 types of the default datasets as follows.

<br/>

- DailyDialog[[2]](#2)
- EmpatheticDialogues[[3]](#3)
- Persona-Chat[[4]](#4)
- BlendedSkillTalk[[5]](#5)

<br/>

You can use whatever data you want, but make sure that you should make `{data_dir}/{train_name}_id.txt` and `{data_dir}/{valid_name}_id.txt` 

consisting of token ids and dialogue split lines.

<img src="https://user-images.githubusercontent.com/16731987/97249049-6ff47080-1846-11eb-81bd-fce6b070db5b.png" alt="The details of the data format." style="width: 80%; margin-left: 0;">

<br/>

If you just run the download & preprocess script, you will also have `{data_dir}/{train_name}.txt` and `{data_dir}/{valid_name}.txt`.

But they are just for checking how the trimmed utterances look like, so they are not used for the actual training.

<br/>

---

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Download & Preprocess all datasets. (If you want to use your own datasets, skip this step.)

   ```shell
   python src/data_load.py --config_path=PATH_TO_CONFIGURATION_FILE
   ```

   - `--config_path`: This indicates the path to the configuration file. (default: `"config.json"`)

   Then there would be `{data_dir}` directory which has corresponding train & validation data files.

   In default setting, the structure of whole data directory should be like below.

   - `data`
     - `train_id.txt`
     - `train.txt`
     - `validation_id.txt`
     - `validation.txt`

   <br/>

3. Run the following command to train the model.

   ```shell
   python src/main.py --config_path=PATH_TO_CONFIGURATION_FILE --mode='train' --ckpt_name=CHECKPOINT_NAME
   ```

   - `--mode`: You have to specify the mode among two options, 'train' or 'inference'.
   - `--ckpt_name`: This specifies the checkpoint file name. If this argument is not specified, then the model would be trained from the beginning and the default checkpoint name becomes `{ckpt_name}`. If you specified the checkpoint name but it does not exist, then the name you put becomes the checkpoint name and the training startes from the beginning. If the name is already that of existing trained model, then the training will be continued starting with that specified checkpoint. (default: `None`)

   <br/>

4. Run below command to conduct an inference with the trained model.

   ```shell
   python src/main.py --config_path=PATH_TO_CONFIGURATION_FILE --mode='inference' --ckpt_name=CHECKPOINT_NAME
   ```

   - `--ckpt_name`: Unlike the case in the training mode, this must specify the name of trained checkpoint which exists.

<br/>

---

### References

<a id="1">[1]</a> How to build a State-of-the-Art Conversational AI with Transfer Learning . (2019, May 9). ([https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313))

<a id="2">[2]</a> Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). Dailydialog: A manually labelled multi-turn dialogue dataset. *arXiv preprint arXiv:1710.03957*. ([https://arxiv.org/abs/1710.03957](https://arxiv.org/abs/1710.03957))

<a id="3">[3]</a> Rashkin, H., Smith, E. M., Li, M., & Boureau, Y. L. (2018). Towards empathetic open-domain conversation models: A new benchmark and dataset. *arXiv preprint arXiv:1811.00207*. ([https://arxiv.org/abs/1811.00207](https://arxiv.org/abs/1811.00207))

<a id="4">[4]</a> Zhang, S., Dinan, E., Urbanek, J., Szlam, A., Kiela, D., & Weston, J. (2018). Personalizing dialogue agents: I have a dog, do you have pets too?. *arXiv preprint arXiv:1801.07243*. ([https://arxiv.org/abs/1801.07243](https://arxiv.org/abs/1801.07243))

<a id="5">[5]</a> Smith, E. M., Williamson, M., Shuster, K., Weston, J., & Boureau, Y. L. (2020). Can You Put it All Together: Evaluating Conversational Agents' Ability to Blend Skills. *arXiv preprint arXiv:2004.08449*. ([https://arxiv.org/abs/2004.08449](https://arxiv.org/abs/2004.08449))