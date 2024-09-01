import json
import torch
from tokenizers import Tokenizer

def remove_unused_merges(merges, vocab):
    return [(a, b) for a, b in [m.split(' ') for m in merges] if all(w in vocab for w in [a, b, a + b])]

def map_special_tokens(data, mapping=None):
    tokens = set()
    if isinstance(data, dict):
        special = data.get('special_tokens')
        if special is not None:
            for v in special.values():
                tokens.update(v['ids'])
                if mapping is not None:
                    v['ids'] = [mapping.get(i) for i in v['ids'] if i in mapping]
    for v in (data.values() if isinstance(data, dict) else data if isinstance(data, list) else []):
        tokens.update(map_special_tokens(v, mapping))
    return tokens

def shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special_tokens):
    assert tokenizer.is_fast
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    assert tokenizer_json['model']['type'] == "BPE"
    if keep_special_tokens:
        keep_indices.update(map_special_tokens(tokenizer_json.get('post_processor')))
    mapping = {old: new for new, old in enumerate(sorted(keep_indices))}
    tokenizer_json['model']['vocab'] = {k: mapping[v] for k, v in tokenizer_json['model']['vocab'].items() if v in mapping}
    tokenizer_json['model']['merges'] = remove_unused_merges(tokenizer_json['model']['merges'], tokenizer_json['model']['vocab'])
    tokenizer_json['added_tokens'] = [{**t, 'id': mapping[t['id']]} for t in tokenizer_json['added_tokens'] if t['id'] in mapping]
    map_special_tokens(tokenizer_json.get('post_processor'), mapping)
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
    return mapping

def shrink_model_embeddings(model, keep_indices, mapping):
    with torch.no_grad():
        row_select = torch.tensor(sorted(keep_indices)).to(model.device)
        new_embed_t = torch.index_select(model.get_input_embeddings().weight.data, 0, row_select)
        new_lm_head = torch.index_select(model.get_output_embeddings().weight.data, 0, row_select)
        model.resize_token_embeddings(len(keep_indices))
        model.get_input_embeddings().weight.data[:] = new_embed_t
        model.get_output_embeddings().weight.data[:] = new_lm_head
        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith('token_id'):
                    setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))

def shrink_embeddings(model, tokenizer, corpus, keep_token_ids=[], keep_model_tokens=True, keep_special_tokens=True):
    keep_indices = set(tokenizer(corpus)['input_ids'])
    keep_indices.update(keep_token_ids)
    if keep_model_tokens:
        for config in [model.config, model.generation_config]:
            for k, v in config.to_dict().items():
                if k.endswith('token_id'):
                    keep_indices.update(v if isinstance(v, list) else [v])
    keep_indices -= set([None])
    mapping = shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special_tokens)
    shrink_model_embeddings(model, keep_indices, mapping=mapping)
    return mapping


##### run_training.py
def run_training(model, tokenizer, formatter, dataset, train_config):
    import torch
    from unsloth import FastLangaugeModel as LanguageModel
    LanguageModel.for_training(model)
    tokenizer.padding_side = 'right'
    model.get_input_embeddings().to(torch.float32)
    model.get_output_embeddings().to(torch.float32)

    from trl import DataCollatorForCompletionOnlyLM
    instruction_masking = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
        **formatter.get_masking_templates()
    )

    from unsloth import UnslothTrainer as Trainer
    from unsloth import UnslothTrainingArguments as TrainingArguments
    from unsloth import is_bfloat16_supported
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=instruction_masking if train_config['instruction_masking'] else None,
        train_dataset=Dataset.from_list(dataset.as_list(formatter)),
        dataset_text_field="text",
        max_seq_length=train_config['max_seq_length'],
        dataset_num_proc=1,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            **train_config['train_args']
        ),
    )
    return trainer.train()
