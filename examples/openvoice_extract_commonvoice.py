import os
import contextlib
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import pandas as pd
from tqdm import tqdm

# TODO: not yet updated to use the wrappers

ckpt_base = 'checkpoints/checkpoints_v2/base_speakers/EN'
ckpt_converter = 'checkpoints/checkpoints_v2/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

base_path = '/data/cv-corpus-21.0-2025-03-14/en'
df = pd.read_csv(f'{base_path}/validated.tsv', sep='\t')
print('number of clips:', len(df))
clients = list(df['client_id'].unique())
print('number of clients:', len(clients))

counter = 0
all_emb = []
all_labels = []
for i, ident in tqdm(enumerate(clients), total=len(clients)):
    # print()
    # print(f'Processing ID {i} of {len(clients)}')
    # print('number of sources:', len(df[df['client_id'] == ident]))
    for source_path in df[df['client_id'] == ident]['path'][:10]:
        try:
            source = f'{base_path}/clips/{source_path}'
            with contextlib.redirect_stdout(None):
                target_se, _ = se_extractor.get_se(source, tone_color_converter,
                                                   target_dir='processed', vad=True)

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
