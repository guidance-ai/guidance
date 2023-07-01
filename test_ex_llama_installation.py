# Tests ExLLama installation
from exllama_lib.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama_lib.tokenizer import ExLlamaTokenizer
from exllama_lib.generator import ExLlamaGenerator
import os, glob

# Directory containing model, tokenizer, generator

model_directory =  "./orca_mini_7B-GPTQ/"

# Locate files we need within that directory

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)[0]

# Create config, model, tokenizer and generator

config = ExLlamaConfig(model_config_path)               # create config from config.json
config.model_path = model_path                          # supply path to model weights file

model = ExLlama(config)                                 # create ExLlama instance and load the weights
tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

cache = ExLlamaCache(model)                             # create cache for inference
generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator

# Configure generator

generator.settings.token_repetition_penalty_max = 1.2
generator.settings.temperature = 0.3
generator.settings.top_p = 0.5
generator.settings.top_k = 100
generator.settings.typical = 0.5


def generate_bias_from_valid_characters(model_config, chars):
    encoded_tokens = [] 
    for char in list(set(chars)):
        encoded_tokens.append(tokenizer.encode(char))

    import torch
    logit_bias = torch.zeros([1, 1, config.vocab_size])

    for encoded_token in encoded_tokens:
        logit_bias[:, :, encoded_token] += 1000.0

    return logit_bias

import pygtrie
tree = pygtrie.CharTrie()

options = ["rogue", "wizard", "fighter"]

# Fill tree with options paths
for option in options:
    for idx in range(len(option)):
            key = option[:idx]
            if tree.has_key(key):
                tree[key].append(option[idx:])
            else:
                tree[key] = [option[idx:]]

first_char_options = []

for option in options:
    first_char_options.append(option[0])

prompts = [
    ("is_rogue", f"""
    Description: You're stealthy and like to steal. You hide well. You walk in the shadows.
    You pickpocket.
    You are a """),

    ("is_fighter", f"""
    Description: You're strong and like to fight. You yield a sword. You wear heavy armor.
    Nothing stops you in a fight.
    You are a """),

    ("is_wizard", f"""
    Description: You cast spells
    Options: {options}
    You are a """),

]

for name, prompt in prompts:
    print("TESTING PROMPT: ", name)
    print (prompt, end = "")

    logit_bias = generate_bias_from_valid_characters(config, first_char_options)
    prefix = ""
    option_fulfilled = False
    max_tokens = 10

    i = 0
    while not option_fulfilled and i < max_tokens:
        prefix += generator.generate_token_with_bias(prompt + prefix, logit_bias=logit_bias)
        suffixes_to_explore = tree[prefix]
        if len(suffixes_to_explore) == 1:
            prefix += suffixes_to_explore[0]
            option_fulfilled = True         
        else:
            valid_chars = []
            for suffix in suffixes_to_explore:
                valid_chars.append(suffix[0])
            logit_bias = generate_bias_from_valid_characters(config, valid_chars)

        i += 1

    print(prefix)
    print("\n\n")