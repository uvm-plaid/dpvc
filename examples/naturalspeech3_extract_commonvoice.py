import os
import contextlib
import torch
import pandas as pd
from tqdm import tqdm

import dpvc

device="cuda:0" if torch.cuda.is_available() else "cpu"
vc_wrapper = dpvc.NaturalSpeech3Wrapper()

base_path = '/data/cv-corpus-21.0-2025-03-14/en'
df = pd.read_csv(f'{base_path}/validated.tsv', sep='\t')
print('number of clips:', len(df))
clients = list(df['client_id'].unique())
print('number of clients:', len(clients))

counter = 0
all_emb = []
all_labels = []
for i, ident in tqdm(enumerate(clients), total=len(clients)):
    for source_path in df[df['client_id'] == ident]['path'][:10]:
        try:
            source = f'{base_path}/clips/{source_path}'
            target_se = vc_wrapper.extract_embedding(source)
            all_emb.append(target_se)
            all_labels.append(torch.tensor(i))
        except Exception as e:
            print(e)
    if i % 1000 == 0:
        all_emb_t = torch.vstack(all_emb)
        all_labels_t = torch.vstack(all_labels)
        print('saving embeddings:', all_emb_t.shape)
        torch.save({'data': all_emb_t, 'labels': all_labels_t}, f'all_emb_labeled_cv_full_{i}.pt')

all_emb_t = torch.vstack(all_emb)
all_labels_t = torch.vstack(all_labels)
print(all_emb_t.shape)
print(all_labels_t.shape)
torch.save({'data': all_emb_t, 'labels': all_labels_t}, 'all_emb_labeled_cv_full.pt')
