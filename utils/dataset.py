# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import random
import torch
import transformers

from typing import Dict
from torch.utils.data import Dataset


IGNORE_TOKEN_ID = -100


def get_dataname(file_name):
    datasets = ["toolbench", "adult", "beijing", "default", "magic", "news", "shoppers"]
    for data in datasets:
        if data in file_name:
            return data
    return None


def process_mlm_masks(inputs, attn_mask, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

    mask_token = tokenizer.mask_token or tokenizer.unk_token
    assert mask_token is not None, "mask token is None, checking tokenizer settings for `mask_token` or `unk_token`"

    masked_indices = torch.bernoulli(torch.full(labels.shape, mlm_probability)).to(torch.bool) & attn_mask.to(torch.bool)
    labels[masked_indices==False] = IGNORE_TOKEN_ID  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with mask_token ([MASK] or [unk])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).to(torch.bool) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).to(torch.bool) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def preprocess_mlm(
    raw_texts,
    encoder_tokenizer,
    decoder_tokenizer,
    use_mlm = False,
    mlm_probability = 0.15
):
    encoded = encoder_tokenizer(
        raw_texts,
        return_tensors="pt",
        padding="max_length",
        max_length=encoder_tokenizer.model_max_length,
        truncation=True,
    )

    encoder_input_ids = encoded.input_ids
    masked_attn_mask = encoded.attention_mask

    if use_mlm:
        masked_encoder_ids, _ = process_mlm_masks(
            encoder_input_ids,
            masked_attn_mask,
            encoder_tokenizer,
            mlm_probability
        )
    else:
        masked_encoder_ids = encoder_input_ids

    # adding eos token
    # decoder_texts = [r + decoder_tokenizer.eos_token for r in raw_texts]
    decoder_texts = raw_texts
    decoder_res = decoder_tokenizer(
        decoder_texts,
        return_tensors="pt",
        padding="max_length",
        max_length=decoder_tokenizer.model_max_length,
        truncation=True,
    )
    decoder_input_ids = decoder_res.input_ids
    decoder_attn_mask = decoder_res.attention_mask
    decoder_labels = torch.where(~decoder_attn_mask.bool(), torch.tensor(IGNORE_TOKEN_ID), decoder_input_ids)

    return dict(
        encoder_input_ids=masked_encoder_ids,
        encoder_attention_mask=masked_attn_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attn_mask,
        decoder_labels=decoder_labels,
    )


def shuffle_json_keys(raw_text):
    try:
        raw_json = json.loads(raw_text)
        assert isinstance(raw_json, dict)
    except:
        return raw_text

    rnd_keys = list(raw_json.keys())
    random.shuffle(rnd_keys)
    rnd_json = {k: raw_json[k] for k in rnd_keys}
    return json.dumps(rnd_json, ensure_ascii=False)


"""
{
"bert_token": [101, 1230, 1947, 1112, 170, 6027, 1110, 5307, 117, 2991, 118, 4606, 1105, 8141, 132, 1119, 3675, 1117, 14861, 107, 1139, 1482, 107, 113, 1380, 1119, 2228, 1146, 1170, 1119, 9340, 11869, 1117, 14861, 1112, 1191, 1152, 1127, 170, 10229, 1554, 1104, 1651, 114, 1105, 1867, 1614, 1176, 107, 1188, 1110, 11968, 13677, 4487, 1643, 117, 1303, 1106, 3999, 1424, 117, 23847, 1105, 4035, 4568, 1424, 1240, 2851, 4568, 2005, 119, 107, 102],
"bert_token_length": 73,
"gpt2_token": [50258, 2399, 3918, 355, 257, 13004, 318, 7209, 11, 2705, 12, 19842, 290, 17144, 26, 339, 3848, 465, 22054, 366, 1820, 1751, 1, 357, 18927, 339, 1838, 510, 706, 339, 14716, 9405, 465, 22054, 355, 611, 484, 547, 257, 15806, 1336, 286, 2444, 8, 290, 1139, 1243, 588, 366, 1212, 318, 21094, 13575, 46670, 11, 994, 284, 6016, 268, 11, 31833, 290, 19128, 268, 534, 3491, 2971, 2250, 526, 50259],
"gpt2_token_length": 70
}
"""
class SupervisedLMDataset(Dataset):
    """Dataset for supervised MLM fine-tuning."""
    def __init__(
        self, raw_data,
        encoder_tokenizer: transformers.PreTrainedTokenizer,
        decoder_tokenizer: transformers.PreTrainedTokenizer,
        shuffle_column = False,
        use_mlm = False,
        mlm_probability = 0.15,
    ):
        super(SupervisedLMDataset, self).__init__()

        if shuffle_column:
            source_texts = [shuffle_json_keys(example["text"]) for example in raw_data]
        else:
            source_texts = [example["text"] for example in raw_data]

        data_dict = preprocess_mlm(source_texts, encoder_tokenizer, decoder_tokenizer, use_mlm, mlm_probability)
        self.encoder_input_ids = data_dict["encoder_input_ids"]
        self.encoder_attention_mask = data_dict["encoder_attention_mask"]
        self.decoder_input_ids = data_dict["decoder_input_ids"]
        self.decoder_attention_mask = data_dict["decoder_attention_mask"]
        self.decoder_labels = data_dict["decoder_labels"]

    def __len__(self):
        return len(self.encoder_input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.encoder_input_ids[i],
            attention_mask=self.encoder_attention_mask[i],
            decoder_ids=self.decoder_input_ids[i],
            decoder_attn_mask=self.decoder_attention_mask[i],
            decoder_labels=self.decoder_labels[i],
        )


class LazySupervisedLMDataset(Dataset):
    """Dataset for supervised MLM fine-tuning."""
    def __init__(
        self, raw_data,
        encoder_tokenizer: transformers.PreTrainedTokenizer,
        decoder_tokenizer: transformers.PreTrainedTokenizer,
        shuffle_column = False,
        use_mlm = False,
        mlm_probability = 0.15,
    ):
        super(LazySupervisedLMDataset, self).__init__()

        if shuffle_column:
            self.source_texts = [shuffle_json_keys(example["text"]) for example in raw_data]
        else:
            self.source_texts = [example["text"] for example in raw_data]

        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.use_mlm = use_mlm
        self.mlm_probability = mlm_probability
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess_mlm(
            [self.source_texts[i], ],
            self.encoder_tokenizer,
            self.decoder_tokenizer,
            use_mlm=self.use_mlm,
            mlm_probability=self.mlm_probability,
        )
        ret = dict(
            input_ids=ret["encoder_input_ids"][0],
            attention_mask=ret["encoder_attention_mask"][0],
            decoder_ids=ret["decoder_input_ids"][0],
            decoder_attn_mask=ret["decoder_attention_mask"][0],
            decoder_labels=ret["decoder_labels"][0],
        )
        self.cached_data_dict[i] = ret
        return ret


if __name__ == "__main__":
    from transformers import AutoTokenizer, LlamaTokenizer

    # tests = [
    #     {"id": 32561, "text": "{\"age\": 25, \"workclass\": \"Private\", \"fnlwgt\": 226802, \"education\": \"11th\", \"education-num\": 7, \"marital-status\": \"Never-married\", \"occupation\": \"Machine-op-inspct\", \"relationship\": \"Own-child\", \"race\": \"Black\", \"sex\": \"Male\", \"capital-gain\": 0, \"capital-loss\": 0, \"hours-per-week\": 40, \"native-country\": \"United-States\", \"income\": \"<=50K\"}", "target_col": "income", "target": "<=50K"},
    #     {"id": 32562, "text": "{\"age\": 38, \"workclass\": \"Private\", \"fnlwgt\": 89814, \"education\": \"HS-grad\", \"education-num\": 9, \"marital-status\": \"Married-civ-spouse\", \"occupation\": \"Farming-fishing\", \"relationship\": \"Husband\", \"race\": \"White\", \"sex\": \"Male\", \"capital-gain\": 0, \"capital-loss\": 0, \"hours-per-week\": 50, \"native-country\": \"United-States\", \"income\": \"<=50K\"}", "target_col": "income", "target": "<=50K"},
    # ]
    tests = [
        {"id": 32561, "text": "Hello word!"},
        {"id": 32562, "text": "Something new..."},
    ]
    enc_tok = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased",
        model_max_length=10,
    )

    if True:
        dec_tok = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            # "meta-llama/Meta-Llama-3.1-8B-Instruct",
            model_max_length=10,
            add_bos_token=True,
            add_eos_token=True,
            padding_side="left",
            use_fast=False,
        )
        dec_tok.pad_token = dec_tok.unk_token
        # dec_tok.pad_token = dec_tok.eos_token
    else:
        dec_tok = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            model_max_length=10,
            add_bos_token=True,
            add_eos_token=True,
            padding_side="left",
        )
        from tokenizers import processors
        bos = dec_tok.bos_token
        eos = dec_tok.eos_token
        dec_tok._tokenizer.post_processor = processors.Sequence(
            [
                processors.ByteLevel(trim_offsets=False),
                processors.TemplateProcessing(
                    single=f"{bos}:0 $A:0 {eos}:0",
                    pair=f"{bos}:0 $A:0 {bos}:1 $B:1 {eos}:1",
                    special_tokens=[
                        (bos, dec_tok.bos_token_id),
                        (eos, dec_tok.eos_token_id),
                    ],
                ),
            ]
        )
        dec_tok.pad_token = dec_tok.eos_token

    ds = LazySupervisedLMDataset(tests, enc_tok, dec_tok, shuffle_column=False)
    print(ds.source_texts)
    for idx, one in enumerate(ds):
        print(f"[Sample {idx}] ***********")
        print(f"[Sample {idx}] Origin Text -> {tests[idx]}")
        print(f"[Sample {idx}] Encoder IDs -> {one['input_ids']}")
        print(f"[Sample {idx}] Encoder msk -> {one['attention_mask']}")
        print(f"[Sample {idx}] Decoder IDs -> {one['decoder_ids']}")
        print(f"[Sample {idx}] Decoder msk -> {one['decoder_attn_mask']}")
        print(f"[Sample {idx}] Decoder lbs -> {one['decoder_labels']}")
        print(f"[Sample {idx}] ***********")
