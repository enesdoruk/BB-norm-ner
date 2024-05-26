from dataset import *
from metrics import *
from preprocessing import *
from tools import *
from utils import * 


from transformers import AutoTokenizer, AutoModel, get_scheduler
import torch
from torch import nn
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")


def main(learning_rate, epochs, max_length, embbed_size, model_name) -> None:
    dd_obt = loader_ontobiotope("data/OntoBiotope_BioNLP-OST-2019.obo")

    dd_habObt = select_subpart_hierarchy(dd_obt, ['OBT:000001', 'OBT:000002'])

    ddd_dataTrain = loader_one_bb4_fold(["data/BB-norm/BioNLP-OST-2019_BB-norm_train"])
    dd_habTrain = extract_data(ddd_dataTrain, l_type=["Habitat", "Phenotype"]) 

    ddd_dataDev = loader_one_bb4_fold(["data/BB-norm/BioNLP-OST-2019_BB-norm_dev"])
    dd_habDev = extract_data(ddd_dataDev, l_type=["Habitat", "Phenotype"])

    ddd_dataTest = loader_one_bb4_fold(["data/BB-norm/BioNLP-OST-2019_BB-norm_test"])
    dd_habTest = extract_data(ddd_dataTest, l_type=["Habitat", "Phenotype"])
    
    dd_BB4habTrain_lowercased = lowercaser_mentions(dd_habTrain)
    dd_BB4habDev_lowercased = lowercaser_mentions(dd_habDev)
    dd_BB4habTest_lowercased = lowercaser_mentions(dd_habTest)

    dd_habObt_lowercased = lowercaser_ref(dd_habObt)

    dd_ref = dd_habObt_lowercased
    dd_train = dd_BB4habTrain_lowercased
    dd_val = dd_BB4habDev_lowercased
    dd_test = dd_BB4habTest_lowercased
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")
    
    model = AutoModel.from_pretrained(model_name).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    X_train, y_train = mk_set(dd_train, dd_ref, tokenizer, max_length, device)
    X_val, y_val = mk_set(dd_val, dd_ref, tokenizer, max_length, device)

    train_set = Dataset(X_train, y_train)
    val_set = Dataset(X_val, y_val)

    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(train_dataloader)

    def cos_dist(t1, t2):
        cos = nn.CosineSimilarity()
        cos_sim = cos(t1, t2)*(-1)
        return cos_sim

    loss_fn = cos_dist
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    model.train()
    for epoch in range(epochs):
        for X, y in train_dataloader: 
            batch_loss = None
            for tokenized_mention, tokenized_label in zip(X, y):
                tokenized_mention = tokenized_mention.to(device)
                tokenized_label = tokenized_label.to(device)
                pred = model(tokenized_mention)[0][:,0] 
                ground_truth = model(tokenized_label)[0][:,0]
                loss = loss_fn(pred, ground_truth) 
                if batch_loss == None:
                    batch_loss = loss.reshape(1,1)
                else:
                    batch_loss = torch.cat((batch_loss, loss.reshape(1,1)), dim=1) 

            batch_loss = torch.mean(batch_loss) 
            batch_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        print(f"Fine-tuning: Epoch n° {epoch}, train loss = {batch_loss.item()}")

        with torch.no_grad():
            for X, y in val_dataloader: 
                batch_loss = None
                for tokenized_mention, tokenized_label in zip(X, y):
                    tokenized_mention = tokenized_mention.to(device)
                    tokenized_label = tokenized_label.to(device)
                    pred = model(tokenized_mention)[0][:,0] 
                    ground_truth = model(tokenized_label)[0][:,0]
                    loss = loss_fn(pred, ground_truth) 
                    if batch_loss == None:
                        batch_loss = loss.reshape(1,1)
                    else:
                        batch_loss = torch.cat((batch_loss, loss.reshape(1,1)), dim=1) 

                batch_loss = torch.mean(batch_loss) 
            print(f"Fine-tuning: Epoch n° {epoch},  val loss = {batch_loss.item()}")
        
    
    dd_predictions = inference(dd_ref, model, dd_test, tokenizer, max_length, device)
    score_BB4_onTest = accuracy(dd_predictions, dd_test)
    print("score_BB4_onDev:", score_BB4_onTest)
        
        
if __name__ == "__main__":
    learning_rate = 1e-5
    epochs = 1
    max_length = 30
    embbed_size = 768
    model_name = 'dmis-lab/biobert-base-cased-v1.2'
    
    main(learning_rate=learning_rate, epochs=epochs, max_length=max_length, \
          embbed_size=embbed_size, model_name=model_name )