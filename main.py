import torch
import torch.optim as optim
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
import json
import os
import re
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import pdb
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from config import args
BATCH_SIZE = 3
UPDATE_INTERVAL= 10
START_EPOCH = 0
BLANK_ID = 1035 # bert_uncased for "_"
MASK_ID = 103 # for BERT
MAX_LEN = 512
SEP_TOKEN = 102
WEIGHT_DECAY = 0
MODEL_NAME = 'bert-large-uncased'

def loadData(folder, suffix=None):
    """loads data as nested dicts/lists"""
    lst = []
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            if 'ipynb' in root:
                continue # jupyter tmp file
            if suffix is None or suffix in root:
                name = os.path.join(root, name)
                with open(name) as f:
                    tmp = json.load(f)
                    lst.append(tmp)
                    if not tmp['options']:
                        raise
    print(folder, suffix, len(lst))
    return lst

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def foo(x):
    x = x.encode('ascii')
    return int.from_bytes(x, byteorder='little')

class ClozeDataset(Dataset):
    """
    the simplest data format: {article, options, answers}
    article and answers zero padded, options -1 padded
    options that contains multiple tokens are truncated
     94 articles longer than 512, articles that are much too long are not discarded here, but will be truncated by my BERT model. The ignored options are filled with A.
     5677 answers contains more than 1 BERT tokens, but only 2 of them cannot be disinguished using the initial token
     for BERT, BLANK_ID should be changed into [MASK]
    """
    def __init__(self, data_list):
        super().__init__()
        self.data = []
        self.meta = []
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # how many answers contain multiple bert tokens?
        cnt = 0
        cnt1 = 0
        # how many cannot be distinguished by the initial token?
        cnt2 = 0
        for item in tqdm(data_list):
            # article
            article = item['article'].lower()
            article = tokenizer.encode(article)
            length = len(article)
            article = torch.tensor(article)
            n_blanks_before = sum(article==BLANK_ID)
            if length > MAX_LEN:
                cnt1 += 1
                article = article[:MAX_LEN]
                article[-1] = SEP_TOKEN
            n_blanks = sum(article==BLANK_ID)
            article = (article * (article!=BLANK_ID).long())+(MASK_ID*(article==BLANK_ID).long())
            
            # answers
            answers = [foo(i) - foo('A') for i in item['answers']][:n_blanks]
            answers = torch.tensor(answers)
            
            # options
            options = [[tokenizer.encode(word)[1:-1] for word in line] for line in item['options']][:n_blanks]
            for i, option in enumerate(options):
                if answers.shape[0]> 0:
                    if len(option[answers[i]])>1:
                        cnt += 1
                        if option[answers[i]] in option[0:answers[i]]+option[answers[i]+1:]:
                            cnt2 += 1
                options[i] = [item[0] for item in option]
            # [0] is [CLS], [-1] is sep
            options = torch.tensor(options)
            self.data.append({"article":article, "options":options, "answers":answers})
            self.meta.append({"n_blanks_before":n_blanks_before, "n_blanks_truncated":n_blanks, "article_length":length})
            
        print("%d answers contains multiple tokens"%(cnt))
        print("%d articles exceeds max length"%(cnt1))
        print("%d answers cannot be decided using the initial token"%(cnt2))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def collate_fn(data_list):
    batch = {}
    max_len = {}
    for key in data_list[0]:
        max_len[key] = 0
        for item in data_list:
            max_len[key] = max(max_len[key], item[key].shape[0])
        lst = [item[key] for item in data_list]
        padding_value = 0
        if key == 'answers':
            padding_value = -1
        batch[key] = pad_sequence(lst, batch_first = True, padding_value = padding_value)
    return batch

class Model(nn.Module):
    def __init__(self,):
        super().__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    def forward(self, article, options, answers=None):
        attention_mask = (article > 99)
        result = self.bert(article, attention_mask = attention_mask, labels=article)
        # we compute our custom loss, so there is no need to set the labels
        _, logit = result[0], result[1]
        
        b, l, dim = logit.shape
        blank_mask = article == MASK_ID
        blank_mask = blank_mask.unsqueeze(-1).expand(*logit.shape)
        logit = torch.masked_select(logit, blank_mask).view(-1, dim)
        
        options = options.view(-1)
        mask = options>0
        options = torch.masked_select(options.view(-1), mask).view(-1, 4)
        # removes the padding options

        if not answers is None:
            answers = answers.view(-1)
            answers = torch.masked_select(answers, answers>=0)
            # removes the padding answers
            index = answers.long().unsqueeze(1)
            answer_token = torch.gather(options, 1, index).view(-1)
            # shape: (n_blanks)
            CE = nn.CrossEntropyLoss(reduction='none')
            loss = CE(input = logit, target = answer_token)
            return loss
        
        else:
            option_score = torch.gather(logit, 1, options)
            prediction = torch.argmax(option_score, dim = 1).view(-1)
            return prediction
     
def test():
    model = Model()
    state_dict = torch.load("bak/83.1", map_location='cpu')
    model.load_state_dict(state_dict['model_dict'])
    model = model.cuda()
    test_lst = loadData('ELE','test')
    test_set = ClozeDataset(test_lst)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = False)
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
    model.cuda()
    model.eval()
    results = {}
    for i, data in enumerate(tqdm(test_loader)):
        result = []
        article, options = data['article'], data['options']
        article, options = article.cuda(), options.cuda()
        prediction = model(article, options)
        for j in range(test_set.meta[i]['n_blanks_before']):
            if j < prediction.shape[0]:
                result.append(chr(ord('A')+prediction[j]))
            else:
                result.append('A')
        results["test%04d"%(i+1)] = result

    with open("results.json", "w") as f:
        json.dump(results, f)
        
""" training"""
torch.cuda.set_device(args.local_rank)
dist.init_process_group("nccl", rank=args.local_rank, world_size=args.world_size)
train_lst = loadData('ELE', 'train') 
val_lst = loadData('ELE', 'dev')
test_lst = loadData('ELE', 'test')
cloth_lst = loadData('CLOTH')
clean_lst = [] 
i = 0
""" remove duplicates"""
for idx, item in enumerate(cloth_lst):
    dup = False
    for j in train_lst+val_lst: # no test from cloth, as expected
        if item['options'] == j['options']:
            dup = True
            break
    if not dup:
        clean_lst.append(item)

train_lst = train_lst + clean_lst
tmp = train_lst[0]
print("%d from cloth"%len(clean_lst))
print(tmp)

with open("train_set", 'rb') as f:
    train_set = pickle.load(f)
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, collate_fn = collate_fn)
val_lst = loadData('ELE', 'dev')
val_set = ClozeDataset(val_lst)
val_loader = DataLoader(val_set, batch_size = 1, shuffle = False, collate_fn=collate_fn)

model = Model()
if START_EPOCH > 0:
    state_dict = torch.load("./CKPT/checkpoint_%d"%(START_EPOCH), map_location='cpu')
    model.load_state_dict(state_dict['model_dict'])
model = model.to(args.local_rank)
#model = nn.DataParallel(model)
model = DDP(model, device_ids=[args.local_rank]) 

no_decay = ["bias", "Norm", "norm"]
grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) ],
        "weight_decay": WEIGHT_DECAY,
    },
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0,
    }
]

if WEIGHT_DECAY > 0:
    optimizer = optim.AdamW(grouped_parameters, lr=5e-5) 
else:
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

if args.local_rank is 0:
    writer = SummaryWriter()

for epoch in range(START_EPOCH, 20):
    model.train()
    for i, data in enumerate(tqdm(train_loader)):
        article, options, answers = data['article'].cuda(), data['options'].cuda(), data['answers'].cuda()
        loss = model(article, options, answers)
        loss = loss.mean()
        loss.backward()
        if i % UPDATE_INTERVAL is 0:
            optimizer.step()
            optimizer.zero_grad()
            if args.local_rank is 0:
                writer.add_scalar('loss', loss.item(), i*BATCH_SIZE+epoch*len(train_set)) 

    if args.local_rank is 0:
        model.eval()
        correct = 0.
        total = 0.
        with torch.no_grad():
            for data in tqdm(val_loader):
                article, options, answers = data['article'].cuda(), data['options'].cuda(), data['answers'].cuda()
                pred = model.module(article, options)
                answers = answers.view(-1)
                answers = torch.masked_select(answers, answers>=0)
                correct += (pred == answers).sum().item()
                total += pred.shape[0]

            writer.add_scalar('eval_acc', correct/total, epoch+1)
        print("epoch %d acc: %f"%(epoch+1, correct/total))
        torch.save({'model_dict': model.module.state_dict(),\
                    'optimizer_dict': optimizer.state_dict(),\
                    'eval_acc': correct/total},\
                   "./CKPT/checkpoint_"+str(epoch+1))