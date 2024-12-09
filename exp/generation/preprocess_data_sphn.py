from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


def preprocess_kg(num_patients, timeOpt):
    # Load sphn kg.
    df = pd.read_csv(f"data/sphn_pc_{timeOpt}_{num_patients}.nt", sep=" ", header=None)
    df.drop(columns=df.columns[-1], axis=1, inplace=True)
    df.columns=['h', 'r', 't']
    node_df = df

    # Map id to entities and relations.
    ent_to_id = {k: v for v, k in enumerate(set(node_df['h']).union(set(node_df['t'])), start=0)}
    rel_to_id = {k: v for v, k in enumerate(set(node_df['r']), start=0)}

    triples = node_df.copy()
    triples["h"] = node_df.h.map(ent_to_id)
    triples["t"] = node_df.t.map(ent_to_id)
    triples["r"] = node_df.r.map(rel_to_id)    

    entity = pd.DataFrame({'id': list(ent_to_id.values()), 'entity': list(ent_to_id)})
    relation = pd.DataFrame({'id': list(rel_to_id.values()), 'relation': list(rel_to_id)})

    # Save triples, entities and relations.
    print("Creating Triples / Entities / Relations.")
    triples.to_csv(f'processed_data/sphn_pc_{timeOpt}_triples_{num_patients}.tsv', sep='\t', index=False, header=None)
    entity.to_csv(f'processed_data/sphn_pc_{timeOpt}_entities_{num_patients}.tsv', sep='\t', index=False, header=None)
    relation.to_csv(f'processed_data/sphn_pc_{timeOpt}_relations_{num_patients}.tsv', sep='\t', index=False, header=None)
    print("Triples / Entities / Relations saved.")

    # Get literals.
    numeric_df = node_df[node_df['r'] == '<http://sphn.org/hasValue>'].copy()
    numeric_values = pd.to_numeric(numeric_df.t.values)
    numeric_df['numeric'] = numeric_values
    numeric_arr = np.zeros((len(entity), 1))
    for i, v in numeric_df.t.items():
        num_id = entity[entity.entity == v].id
        numeric_arr[num_id] = numeric_df.numeric.loc[i]

    if timeOpt == 'NT':
        np.save(f"processed_data/sphn_pc_NT_numeric_{num_patients}.npy", numeric_arr)
        print("Literals NT saved.")
    elif timeOpt == 'TR':
        np.save(f"processed_data/sphn_pc_TR_numeric_{num_patients}.npy", numeric_arr)
        print("Literals TR saved.")
    elif timeOpt == 'TS':
        time_df = node_df[node_df['r'].str.contains('<http://sphn.org/hasStartDateTime>|<http://sphn.org/hasDeterminationDateTime>')].copy()
        time_df['sec'] = time_df.t.str.removesuffix('^^<http://www.w3.org/2001/XMLSchema#dateTime>')
        times = []
        for i, t in time_df.sec.items():
            time = datetime.strptime(t, '%Y-%m-%dT%H:%M:%S') - datetime(2020,1,1)
            times.append(time.total_seconds())
        time_df['sec'] = times
            
        qt = QuantileTransformer(n_quantiles=10, random_state=0)
        qt_times = qt.fit_transform(time_df.sec.values.reshape(-1,1))
        time_df['sec'] = list(qt_times.reshape(-1,))
        for i, v in time_df.t.items():
            num_id = entity[entity.entity == v].id
            numeric_arr[num_id] = time_df.sec.loc[i]
        
        np.save(f"processed_data/sphn_pc_TS_numeric_{num_patients}.npy", numeric_arr)
        print("Literals TS saved.")
    elif timeOpt == 'TS_TR':
        time_df = node_df[node_df['r'].str.contains('<http://sphn.org/hasStartDateTime>|<http://sphn.org/hasDeterminationDateTime>')].copy()
        time_df['sec'] = time_df.t.str.removesuffix('^^<http://www.w3.org/2001/XMLSchema#dateTime>')
        times = []
        for i, t in time_df.sec.items():
            time = datetime.strptime(t, '%Y-%m-%dT%H:%M:%S') - datetime(2020,1,1)
            times.append(time.total_seconds())
        time_df['sec'] = times
            
        qt = QuantileTransformer(n_quantiles=10, random_state=0)
        qt_times = qt.fit_transform(time_df.sec.values.reshape(-1,1))
        time_df['sec'] = list(qt_times.reshape(-1,))
        for i, v in time_df.t.items():
            num_id = entity[entity.entity == v].id
            numeric_arr[num_id] = time_df.sec.loc[i]
        np.save(f"processed_data/sphn_pc_TS_TR_numeric_{num_patients}.npy", numeric_arr)
        print("Literals TS TR saved.")