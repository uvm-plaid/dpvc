import os
import contextlib
import torch
import dpvc
import pandas as pd
from tqdm import tqdm


vc_wrapper = dpvc.OpenVoiceWrapper()

base_path = '/data/cv-corpus-21.0-2025-03-14/en'
df = pd.read_csv(f'{base_path}/validated.tsv', sep='\t')
print(df.columns)
print(df['age'].value_counts(dropna=False))
print(df['gender'].value_counts(dropna=False))
print(df['accents'].value_counts(dropna=False))
print(df['variant'].value_counts(dropna=False))
print(df['locale'].value_counts(dropna=False))
print(df['accents'].value_counts(dropna=False).head(20))

print('number of clips:', len(df))
clients = list(df['client_id'].unique())
print('number of clients:', len(clients))

genders = {'male_masculine': -1,
           'female_feminine': 1}
ages = {'teens': -1,
        'twenties': -.6,
        'thirties': -.2,
        'fourties': .2,
        'fifties': .6,
        'sixties': 1}
accents = {'United States English': -1,
           'England English': -.25,
           'Australian English': .25,
           'India and South Asia (India, Pakistan, Sri Lanka)': 1,
           }

counter = 0
all_emb = []
all_ids = []
all_ages = []
all_genders = []
all_accents = []

grp = df.groupby('client_id').head(10)
filtered = grp[(grp['age'].isin(ages)) & (grp['gender'].isin(genders)) & (grp['accents'].isin(accents))]
fgroup = filtered.groupby('client_id').head(1)
print('This many utterances have all features:', len(filtered))
print('This many utterances from unique speakers:', len(fgroup))
print('Accent results:')
print(fgroup['accents'].value_counts())

for i, tup in tqdm(enumerate(fgroup.itertuples(index=False)), total=len(fgroup)):
    source_path = tup.path
    try:
        source = f'{base_path}/clips/{source_path}'
        with contextlib.redirect_stdout(None):
            target_se = vc_wrapper.extract_embedding(source)

        all_emb.append(target_se)
        all_ids.append(torch.tensor(i))
        all_ages.append(torch.tensor(ages[tup.age]))
        all_genders.append(torch.tensor(genders[tup.gender]))
        all_accents.append(torch.tensor(accents[tup.accents]))
    except Exception as e:
        #print(e)
        pass

    if (i+1) % 1000 == 0:
        all_emb_t = torch.vstack(all_emb)
        all_ids_t = torch.vstack(all_ids)
        all_ages_t = torch.vstack(all_ages)
        all_genders_t = torch.vstack(all_genders)
        all_accents_t = torch.vstack(all_accents)
        print('saving embeddings:', all_emb_t.shape)
        #torch.save({'data': all_emb_t, 'ids': all_ids_t, 'ages': all_ages_t, 'genders': all_genders_t, 'accents': all_accents_t}, f'all_emb_labeled_cv_full_{i}.pt')
        # print({'data': all_emb_t, 'ids': all_ids_t, 'ages': all_ages_t, 'genders': all_genders_t})

all_emb_t = torch.vstack(all_emb)
all_ids_t = torch.vstack(all_ids)
all_ages_t = torch.vstack(all_ages)
all_genders_t = torch.vstack(all_genders)
all_accents_t = torch.vstack(all_accents)
print('saving embeddings:', all_emb_t.shape)
torch.save({'data': all_emb_t, 'ids': all_ids_t, 'ages': all_ages_t, 'genders': all_genders_t, 'accents': all_accents_t}, f'all_emb_labeled_cv_full_{i}.pt')

