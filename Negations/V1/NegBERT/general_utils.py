import chardet
import torch


def get_encoding_type(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def pad_sequences(sequences, maxlen, value=0, padding='post', truncating='post', dtype=torch.long):
    num_sequences = len(sequences)
    out_seqs = torch.full((num_sequences, maxlen), value, dtype=dtype)

    for i, seq in enumerate(sequences):
        seq_tensor = torch.tensor(seq, dtype=dtype)

        len_seq = len(seq)
        if len_seq > maxlen:
            if truncating == 'post':
                seq_tensor = seq_tensor[:maxlen]
            elif truncating == 'pre':
                seq_tensor = seq_tensor[-maxlen:]
        elif len_seq < maxlen:
            if padding == 'post':
                seq_tensor = torch.cat([seq_tensor, torch.full((maxlen - len_seq,), value, dtype=dtype)])
            elif padding == 'pre':
                seq_tensor = torch.cat([torch.full((maxlen - len_seq,), value, dtype=dtype), seq_tensor])

        out_seqs[i] = seq_tensor

    return out_seqs
