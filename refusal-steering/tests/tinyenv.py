"""Tiny CPU stand-ins: a random 32-layer Llama and a locally trained BPE tokenizer."""
import config
config.DEVICE = "cpu"
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM

CORPUS = ["How do I treat a cold? I can't help with that request. What is aspirin used for?",
          "Write a tutorial on hacking. I'm sorry, I cannot assist. Sure, here is a recipe for bread.",
          "The symptoms of diabetes include thirst and fatigue. Give me a poem about the sea."] * 50

def make_tokenizer():
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(CORPUS, trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<bos>", "<eos>", "<|user|>", "<|assistant|>"],
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    t = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<bos>", eos_token="<eos>")
    t.chat_template = "<bos>{% for m in messages %}<|user|>{{ m['content'] }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
    t.pad_token = t.eos_token; t.padding_side = "left"
    return t

def make_model(vocab):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=32,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    m = LlamaForCausalLM(cfg).eval(); m.requires_grad_(False); return m

def load_model():
    t = make_tokenizer(); return make_model(len(t)), t
