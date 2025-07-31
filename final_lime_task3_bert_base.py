args = {}
args["input_model_path"] = "google-bert/bert-base-uncased"

import torch
from tqdm import tqdm
import time
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import os
import gc
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
import numpy as np
import time

MAX_LEN = 512

def tokenize_sent(sentence, tokenizer):

    tokenized_sentence = []
    sentence = str(sentence).strip()

    for word in sentence.split():
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)

    return tokenized_sentence



input_path = './'
num_labels = 2
BATCH_SIZE = 32

class LossFunction(nn.Module):
    def forward(self, probability):
        loss = torch.log(probability)
        loss = -1 * loss
        # print(loss)
        loss = loss.mean()
        # print(loss)
        return loss



class MainModel(BertPreTrainedModel):
    def __init__(self, config, loss_fn = None):
        super(MainModel,self).__init__(config)
        self.num_labels = num_labels
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained(args["input_model_path"],config = config)
        self.classifier = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels,device):

        output = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        output = output.last_hidden_state
        output = output[:,0,:]
        classifier_out = self.classifier(output)
        main_prob = F.softmax(classifier_out, dim = 1)
        main_gold_prob = torch.gather(main_prob, 1, labels)
        loss_main = self.loss_fn.forward(main_gold_prob)
        return loss_main,main_prob

tokenizer = AutoTokenizer.from_pretrained(args["input_model_path"])
model = MainModel.from_pretrained(args["input_model_path"], loss_fn = LossFunction())

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)
model.to(device)
print(model)

df = pd.read_csv("questions.csv")
df = df.head(15000)
#df.isna().sum()

df = df.dropna(subset = ["question1","question2"])

#df.isna().sum()

texts = []

for idx,item in df["question1"].items():
    texts.append(df["question1"][idx] + "  " + df["question2"][idx])

print(texts[0])
print(len(texts))

# create prediction function

max_len = 512

def predict_f(texts_old, batch_size=32):
    all_predictions = []
    for i in range(0, len(texts_old), batch_size):
        texts = texts_old[i:i+batch_size]
        f_inputs = {}
        f_inputs["target"] = []
        f_inputs['token_type_ids'] = []
        f_inputs['mask'] = []
        f_inputs['ids'] = []
        for text in texts:
            if "  " in text:
                sent1 = text.split("  ")[0]
                sent2 = text.split("  ")[1]
            else:
                sent1 = text
                sent2 = ""

            if len(sent1)>0 and sent1[len(sent1)-1] == " ":
                sent1 = sent1[:-1]
            if len(sent2)>0 and sent2[0] == " ":
                sent2 = sent2[1:]

            label = 0
            target = []
            target.append(label)

            token_type_ids = []
            token_type_ids.append(0)
            sent1 = tokenize_sent(sent1,tokenizer)
            sent2 = tokenize_sent(sent2,tokenizer)
            for i in enumerate(sent1):
                token_type_ids.append(0)
            token_type_ids.append(1)
            for i in enumerate(sent2):
                token_type_ids.append(1)
            token_type_ids.append(1)
            input_sent = ['[CLS]'] + sent1 + ['[SEP]'] + sent2 + ['[SEP]']
            input_sent = input_sent + ['[PAD]' for _ in range(max_len - len(input_sent))]
            token_type_ids = token_type_ids + [0 for _ in range(max_len - len(token_type_ids))]
            attn_mask = [1 if tok != '[PAD]' else 0 for tok in input_sent]
            ids = tokenizer.convert_tokens_to_ids(input_sent)

            f_inputs["target"].append(target)
            f_inputs['token_type_ids'].append(token_type_ids)
            f_inputs['mask'].append(attn_mask)
            f_inputs['ids'].append(ids)

        f_inputs["target"] = torch.tensor(f_inputs["target"], dtype=torch.long)
        f_inputs['token_type_ids'] = torch.tensor(f_inputs['token_type_ids'], dtype=torch.long)
        f_inputs['mask'] = torch.tensor(f_inputs['mask'], dtype=torch.long)
        f_inputs['ids'] = torch.tensor(f_inputs['ids'], dtype=torch.long)

        input_ids = f_inputs['ids'].to(device, dtype=torch.long)
        mask = f_inputs['mask'].to(device, dtype=torch.long)
        targets = f_inputs['target'].to(device, dtype=torch.long)
        token_type_ids = f_inputs['token_type_ids'].to(device, dtype = torch.long)

        with torch.no_grad():
            loss_main,main_prob = model(input_ids=input_ids, attention_mask=mask, token_type_ids = token_type_ids, labels=targets, device = device)
            for i in main_prob.cpu():
                all_predictions.append(i)
    all_predictions = np.array(all_predictions)

    return all_predictions

predictions = []
sent1 = []
sent2 = []
true_out = []
nd_words = []
d_words = []
output_prob = []

def give_class (output_tensor):
    class_labels = ["not duplicate", "duplicate"]
    predicted_class_idx = np.argmax(output_tensor)
    predicted_class = class_labels[predicted_class_idx]
    return predicted_class

def sort_list (x):
    #sorted_list = sorted(x, key=lambda x: x[1], reverse=True)
    sorted_list = sorted(x)
    return sorted_list

class_names=["not duplicate", "duplicate"]

from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=["not duplicate", "duplicate"])

idx = 0
begin = time.time()
for item in texts:
    texts = [item]
    predicted_out = predict_f(texts)
    out_pred_arr = []
    out_pred_arr.append(predicted_out[0][0].item())
    out_pred_arr.append(predicted_out[0][1].item())
    output_prob.append(out_pred_arr)
    predictions.append(give_class(predicted_out[0]))
    sent1.append(df["question1"][idx])
    sent2.append(df["question2"][idx])
    true_out.append(df["is_duplicate"][idx])
    explanation = explainer.explain_instance(item, predict_f, num_features=20, num_samples=500)
    #explanation.local_exp = {k: torch.tensor(v).to(device) for k, v in explanation.local_exp.items()}
    lime_values = explanation.as_list()
    #explanation.show_in_notebook(text=item)

    nd_arr = []
    d_arr = []

    #print('Model Prediction:', df["is_duplicate"][idx])

    probs = predict_f([item])
    #print("LIME Probabilities for 'non-duplicate' and 'duplicate'", probs[0])

    #print('LIME Values:', lime_values)
    #print("---------------------------------------------------")

    for word, weight in lime_values:
    	if (weight > 0):
    		d_arr.append(word)
    	elif (weight < 0):
    		nd_arr.append(word)

    d_words.append(sort_list(d_arr))
    nd_words.append(sort_list(nd_arr))
    
    idx+=1
    print(idx)
    #print("Words contributing to Duplicate", d_arr)
    #print("Words contributing to Non-Duplicate", nd_arr)
	    
final_result = pd.DataFrame({
    "sent1":sent1,
    "sent2":sent2,
    "true_out":true_out,
    "d_words":d_words,
    "nd_words":nd_words,
    "predicted_out":predictions,
    "predicted_prob":output_prob
})

end = time.time() 

# total time taken 
print(f"Total runtime of the program is {end - begin}")

final_result.to_csv("/home/rohitraj/shap_QQP/lime/lime_outputs/QQP_pred_bert_base_pretrained.csv")
