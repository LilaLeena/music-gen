hparams = {
    "n_layer": 4, 
    "n_head": 4, # How many attention heads to use -> at how many "things" in the past the net can effectively look at (as the softmaxed attention mostly brings up one value)
    "embedding_dim": 32, # The dimension of the token embedding. In the original paper, a 50k dictionary was embedded into 768 dims. As we have a far smaller dictionary, we can do with far less
    "max_seq_length": 2048, # The longest piece we can process, everything longer will be discarted
    "lr": 1e-4, # Too high -> diverge, too low -> slow training
    "batch_size": 16, # The bigger the better
    "epoch_length": 10000, # Checkpoint every epoch
    "epochs": 100,
    "checkpoint_dir": "checkpoints",
    "load_checkpoint": None,
    "use_gpu": True,
    "use_multi_gpu": True,
    "download_url": "https://raw.githubusercontent.com/adactio/TheSession-data/master/json/tunes.json",
}
hparams["iters"] = hparams["epochs"] * hparams["epoch_length"]

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import transformers
from tqdm import tqdm

def tensor_to_abc(dictionary, data):
    return ''.join([dictionary[x] for x in data if x != -100 and x != 0 and x != 1])

def abc_to_list(dictionary, text, bos=True, eos=True):
    return ([dictionary.index('<bos>')] if bos else []) + [dictionary.index(char) for char in text] + ([dictionary.index('<eos>')] if eos else [])

def download_data(hparams):
    r = requests.get(hparams['download_url'])
    assert r.status_code == 200
    data = pd.DataFrame(json.loads(r.text))

    # Strip all whitespaces from abc
    data['abc'] = data['abc'].map(lambda text: text.replace(" ", "").replace("\r", "").replace("\n", "").replace("\t", "").replace("\x14", "").replace("\x1a", "").replace("\xa0", "").replace(u"\u2028", ""))

    dictionary = list(data['abc'].map(lambda text: [char for char in text]))
    dictionary = [item for sublist in dictionary for item in sublist]
    dictionary = ['<bos>', '<eos>'] + list(np.unique(dictionary))


    features = data['abc'].map(lambda text:  abc_to_list(dictionary, text))
    features = [x for x in features if len(x) <= hparams['max_seq_length']]
    features.sort(reverse=True, key=lambda x: len(x))
    features = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in features], batch_first=True, padding_value=-100)

    return features, dictionary


def create_model(hparams, dictionary):
    # Config docs: https://huggingface.co/transformers/model_doc/gpt2.html#gpt2config
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size = len(dictionary),
        n_embd = hparams["embedding_dim"],
        n_layer = hparams["n_layer"],
        n_head = hparams["n_head"],
        n_positions = hparams['max_seq_length'],
        n_ctx = hparams['max_seq_length']
    ))

    if hparams["load_checkpoint"]:
        model.load_state_dict(torch.load(hparams["load_checkpoint"], map_location=lambda storage, location: storage))

    if hparams["use_multi_gpu"]:
        assert torch.cuda.device_count() > 1
        print("Using %d GPUs" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    optim = torch.optim.Adam(model.parameters(), lr=hparams["lr"])

    return model, optim

def synthesize(dictionary, starting_sequence):
    model.eval()
    with torch.no_grad():
        starting_sequence_tensor = torch.tensor(abc_to_list(dictionary, starting_sequence, eos=False)).unsqueeze(0).to(device)
        # A nice article about the parameters https://huggingface.co/blog/how-to-generate
        pred = model.generate(
            starting_sequence_tensor,
            max_length=100,
            min_length=20,
            do_sample=True,
            temperature=0.8,
            top_k=15,
            top_p=0.9,
            eos_token_id=dictionary.index('<eos>'),
            bos_token_id=dictionary.index('<bos>'))
        return tensor_to_abc(dictionary, pred.cpu()[0])

def train(hparams, model, optim, features):
    model.train()
    for i in tqdm(range(0, hparams['iters'])):
        indices = np.random.choice(len(features), hparams['batch_size'])
        batch = features[indices].clone()
        mask = batch == -100
        batch[mask] = 0
        
        optim.zero_grad()
        # The forward function is documented here: https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
        loss, predictions, past = model(batch, attention_mask=mask, labels=features[indices])
        if multigpu:
            loss = loss.mean()
        loss.backward()
        optim.step()
        
        if i % hparams['epoch_length'] == 0:
            tqdm.write('Epoch %d, loss %f' % (i // hparams['epoch_length'], loss.cpu()))

            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()
            torch.save(state_dict, '%s/model%d.pth' % (hparams['checkpoint_dir'], i//hparams['epoch_length']))


    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    torch.save(state_dict, '%s/final.pth' % hparams['checkpoint_dir'])

if __name__ == '__main__':
    
    features, dictionary = download_data(hparams)
    model, optim = create_model(hparams, dictionary)
    
    if hparams["use_gpu"]:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        features = features.to(device) # The entire dataset fits on a gpu easily
    
    train(hparams, model, optim, features)

