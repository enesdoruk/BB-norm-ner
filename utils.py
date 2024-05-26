from scipy.spatial.distance import cdist

import torch
from tqdm import tqdm
import numpy as np


def tokenize(sentence, tokenizer, max_length, device):
    return tokenizer.encode(sentence, padding="max_length", max_length=max_length, truncation=True, add_special_tokens = True, return_tensors="pt").to(device) # Tokenize input into ids.

def mk_set(dataset, concept_dict, tokenizer, max_length, device):
    # Constructs two dictionnaries containing tokenized mentions (X) and associated labels (Y) respectively.
    X = dict()
    y = dict()
    for i, id in enumerate(dataset.keys()):
        X[i] = tokenizer.encode(dataset[id]['mention'], padding="max_length", max_length=max_length, truncation=True, add_special_tokens = True, return_tensors="pt")
        y[i] = tokenizer.encode(concept_dict[dataset[id]['cui'][0]]['label'], padding="max_length", max_length=max_length, truncation=True, add_special_tokens = True, return_tensors="pt")
    nbMentions = len(X.keys())
    print("Number of mentions:", nbMentions)
    return X, y

def inference(dd_ref, model, dd_test, tokenizer, max_length, device):
    nbLabtags = 0
    dd_conceptVectors = dict()
    embbed_size = None
    with torch.no_grad():
        for cui in tqdm(dd_ref.keys(), desc='Building embeddings from ontology labels'):
            dd_conceptVectors[cui] = dict()
            dd_conceptVectors[cui][dd_ref[cui]["label"]] = model(tokenize(dd_ref[cui]['label'], tokenizer, max_length, device))[0][:,0].cpu().detach().numpy()
            nbLabtags += 1
            if embbed_size == None:
                embbed_size = len(dd_conceptVectors[cui][dd_ref[cui]["label"]][0])
            if dd_ref[cui]["tags"]:
                for tag in dd_ref[cui]["tags"]:
                    nbLabtags += 1
                    dd_conceptVectors[cui][tag] = model(tokenize(tag, tokenizer, max_length, device))[0][:,0].cpu().detach().numpy()


    X_pred = np.zeros((len(dd_test.keys()), embbed_size))
    with torch.no_grad():
        for i, id in tqdm(enumerate(dd_test.keys()), desc ='Building embeddings from test labels'):
            tokenized_mention = torch.tensor(tokenize(dd_test[id]['mention'], tokenizer, max_length, device).to(device))
            X_pred[i] = model(tokenized_mention)[0][:,0].cpu().detach().numpy()

  
    dd_predictions = dict()
    for id in dd_test.keys():
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []

    labtagsVectorMatrix = np.zeros((nbLabtags, embbed_size))
    i = 0
    for cui in dd_conceptVectors.keys():
        for labtag in dd_conceptVectors[cui].keys():
            labtagsVectorMatrix[i] = dd_conceptVectors[cui][labtag]
            i += 1

    scoreMatrix = cdist(X_pred, labtagsVectorMatrix, 'cosine') 

    i=0
    for i, id in enumerate(dd_test.keys()):
        minScore = min(scoreMatrix[i])
        j = -1
        stopSearch = False
        for cui in dd_conceptVectors.keys():
            if stopSearch == True:
                break
            for labtag in dd_conceptVectors[cui].keys():
                j += 1
                if scoreMatrix[i][j] == minScore:
                    dd_predictions[id]["pred_cui"] = [cui]
                    stopSearch = True
                    break
    del dd_conceptVectors
    return dd_predictions


def generate_predict(dd_pred, dd_resp):
    result = []
    for id in dd_resp.keys():
        result.append((dd_resp[id]['T'], dd_resp[id]['type'], dd_resp[id]['mention'], dd_pred[id]["pred_cui"][0]))
    
    import pdb; pdb.set_trace()
    
    output_filename = 'output.a2'
    with open(output_filename, 'w') as file:
        for sublist in result:
            file.write(' '.join(map(str, sublist)) + '\n')