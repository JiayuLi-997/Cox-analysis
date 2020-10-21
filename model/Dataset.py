import pandas as pd
import pickle
import numpy as np
import random
from sklearn.utils import check_random_state
import gc
import json

class Vocabulary(object):
    def __init__(self, token_to_idx=None, padding={"<PAD>": 0}):

        # Token to index
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx
        
        # Index to token
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        
        if padding:
            padstr, self.padding_idx = list(padding.items())[0]
            if self.padding_idx in self.idx_to_token or padstr in self.token_to_idx:
                if self.token_to_idx[padstr] != self.padding_idx:
                    print ("PAD idx/token already in vocabulary")
            self.token_to_idx[padstr] = self.padding_idx
            self.idx_to_token[self.padding_idx] = padstr

    def to_serializable(self):
        return {'token_to_idx': self.token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def add_tokens(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        return self.token_to_idx[token]

    def lookup_index(self, index):
        if index not in self.idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self.idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self.token_to_idx)
    
class SequenceVectorizer(object):
    def __init__(self, vocab):
        self.vocab = vocab
        self.num_tokens = len(self.vocab.token_to_idx)

    def vectorize(self, token_seq, return_lens=True):
        """from token sequence to idx sequence"""
        idx_seq = [self.vocab.lookup_token(token) for token in token_seq]
        if return_lens:
            return idx_seq, len(idx_seq)
        else:
            return idx_seq
        
    def vectorize_deep(self, token_seq, return_lens=True):
        """from token sequence to idx sequence"""
        idx_seq = [[self.vocab.lookup_token(token) for token in tokens] for tokens in token_seq]
        if return_lens:
            return idx_seq, len(idx_seq)
        else:
            return idx_seq
    
    def unvectorize(self, idx_seq, remove_pad=True):
        """from idx sequence to token sequence"""
        if remove_pad:
            return [self.vocab.lookup_index(index) for index in idx_seq if index != self.vocab.padding_idx]
        return [self.vocab.lookup_index(index) for index in idx_seq]
    
    @classmethod
    def padding(cls, seq, max_length, padding=0):
        if max_length is not None and max_length > 0:
            if len(seq) >= max_length:
                return seq[:max_length]
            else:
                return seq + [padding for _ in range(max_length - len(seq))]
        assert False
    
    @classmethod
    def padding_deep(cls, seq, max_length1, max_length2, padding=0):
        if max_length1 is not None and max_length1 > 0:
            if len(seq) >= max_length1:
                return [SequenceVectorizer.padding(tseq, max_length2) for tseq in seq[:max_length1]]
            else:
                return [SequenceVectorizer.padding(tseq, max_length2) for tseq in seq] + [[padding for __ in range(max_length2)] for _ in range(max_length1 - len(seq))]
        assert False
    
    @classmethod
    def from_corpus(cls, corpus):
        tokens = set()
        for tseq in corpus:
            for taction in tseq:
                if type(taction) == list:
                    for ttoken in taction:
                        tokens.add(ttoken)
                else:
                    tokens.add(taction)
        vocab = Vocabulary()
        vocab.add_tokens(tokens)        
        return cls(vocab)

    @classmethod
    def from_serializable(cls, contents):
        vocab = Vocabulary.from_serializable(contents['vocab'])
        return cls(vocab = vocab)
    
    def to_serializable(self):
        return {'vocab': self.vocab.to_serializable()}

class SequenceDataset:
    
    def __init__(self, df=None, vectorizer_dict={}, state="token", idx_oversampled=None):
        
        self.df = pd.DataFrame() if df is None else df
        self.vectorizer_dict = vectorizer_dict
        self.idx_oversampled = idx_oversampled
        self.state = state
        
        self.__update_info()
        
    def filtering(self, remain_idxs):
        
        self.df = self.df.iloc[remain_idxs,:].copy()
        self.df.reset_index(drop=True, inplace=True)
        self.__update_info()
        
    def __update_info(self):
        """self.length; self.df.seq_length"""
        if self.df is not None:
            self.length = len(self.df)
            seqkeys = list(self.vectorizer_dict.keys())
            if len(self.vectorizer_dict) > 0 and "seq_length" not in self.df.columns:
                self.df['seq_length'] = self.df[seqkeys[0]].map(lambda t: len(t))
        else:
            self.length = 0
    
    def drop_by_length(self, min_seqlen=None, max_seqlen=None, inplace=True):
        if self.df is None or (min_seqlen is None and max_seqlen is None):
            return
        
        valid_idx = np.array([True for _ in range(self.length)])
        if min_seqlen is not None:
            valid_idx = valid_idx & (self.df['seq_length'] >= min_seqlen)
        if max_seqlen is not None:
            valid_idx = valid_idx & (self.df['seq_length'] <= max_seqlen)
        self.df = self.df[valid_idx].copy()
        self.df.reset_index(drop=True, inplace=True)
        gc.collect()
        self.__update_info()
        
    def __valid_sequence(self, key, sequence):
        if key in self.df.columns:
            print ("{} already in the dataset".format(key))
        if self.length != 0 and self.length != len(sequence):
            print ("Length doesn't match")
            return False
        return True
    
    def add_numerical_data(self, key, ndata):
        assert self.__valid_sequence(key, ndata)
        self.df[key] = ndata
        self.__update_info()
    
    def add_sequence_and_make_vectorizer(self, key, sequence, **kwargs):
        assert self.__valid_sequence(key, sequence)
        self.df[key] = sequence
        self.vectorizer_dict[key] = SequenceVectorizer.from_corpus(sequence, **kwargs)
        self.__update_info()
        
    def add_sequence_and_load_vectorizer(self, key, sequence, vectorizer_filepath):
        assert self.__valid_sequence(key, sequence)
        self.df[key] = sequence
        self.vectorizer_dict[key] = self.load_vectorizer(vectorizer_filepath, single=True)
        self.__update_info()
    
    def dump(self, pkl_filepath):
        pickle.dump({"df": self.df, 
                     "vectorizer": dict([(tk, tvec.to_serializable()) for tk, tvec in self.vectorizer_dict.items()]),
                     "state": self.state,
                     "idx_oversampled": self.idx_oversampled
                    },
                     open(pkl_filepath, "wb"))
    
    def dump_chunk(self, pkl_addr):
        pass
    
    @classmethod
    def load(cls, pkl_filepath):
        data = pickle.load(open(pkl_filepath, "rb"))
        return cls(data['df'], 
                   dict([(tk, SequenceVectorizer.from_serializable(tvec)) for tk, tvec in data['vectorizer'].items()]),
                   data['state'],
                   data['idx_oversampled']
                  )
    
    @classmethod
    def load_chunk(cls, pkl_addr):
        pass
    
    def load_vectorizer(self, vectorizer_filepath, single=False):
        with open(vectorizer_filepath) as fp:
            temp = json.load(fp)
        if single:
            return SequenceVectorizer.from_serializable(temp)
        self.vectorizer_dict = dict([(tk, SequenceVectorizer.from_serializable(tvec)) for tk, tvec in temp.items()])
        
    def save_vectorizer(self, vectorizer_dict_filepath):
        with open(vectorizer_dict_filepath, "w") as fp:
            json.dump(dict([(tk, tvec.to_serializable()) for tk, tvec in self.vectorizer_dict.items()]), fp)
    
    def __str__(self):
        return "<Dataset(keys=[{}], size={})".format(", ".join(self.df.columns), self.length)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        ans = {}
        for tkey in self.vectorizer_dict:
            ans[tkey] = self.vectorizer_dict[tkey].vectorize(row[tkey])
        return ans
        
    def vectorize_all(self):
        if self.state == "idx":
            print ("already in idx version")
            return 
        for tkey, tvec in self.vectorizer_dict.items():
            if tkey in ['doc_kt_seq', 'doc_ttseg_seq']:
                self.df[tkey] = self.df[tkey].map(lambda t: tvec.vectorize_deep(t, return_lens=False))
            else:
                self.df[tkey] = self.df[tkey].map(lambda t: tvec.vectorize(t, return_lens=False))
        self.state = "token"
    
    def unvectorize_all(self):
        if self.state == "token":
            print ("already in token version")
            return 
        for tkey, tvec in self.vectorizer_dict.items():
            self.df[tkey] = self.df[tkey].map(tvec.unvectorize)
        self.state = "idx"
    
    def over_sampling(self, y, ratio):
        self.idx_oversampled = self.__over_sampling(y, ratio)

    
    def to_numpy(self, keys=None, idxs=None, padding=True, max_length=None, over_sampling=None, max_length2=20):
        if keys is None:
            keys = list(self.df.columns)
        if idxs is None:
            idxs = np.arange(self.length)
        
        if over_sampling is not None and self.idx_oversampled is not None:
            idxs = idxs[self.idx_oversampled]

        ans = {}
        for tkey in keys:
            if tkey not in self.vectorizer_dict:
                assert tkey in self.df.columns
                ans[tkey] = np.array(self.df.iloc[idxs,:][tkey].values)
            else:
                if padding and max_length is not None:
                    if tkey in ['doc_kt_seq', 'doc_ttseg_seq']:
                        ans[tkey] = np.array([np.array(SequenceVectorizer.padding_deep(t, max_length, max_length2, 0)) for t in self.df.iloc[idxs,:][tkey].values])
                    else:
                        ans[tkey] = np.array([np.array(SequenceVectorizer.padding(t, max_length, 0)) for t in self.df.iloc[idxs,:][tkey].values])
                else:
                    ans[tkey] = np.array([np.array(t) for t in self.df.iloc[idxs,:][tkey].values])
        return ans
    
    def train_test_split(self, valid_ratio=0.15, test_ratio=0.15, random_state=0):
        
        all_size = self.length
        idxs = np.arange(all_size)
        random.seed(random_state)
        random.shuffle(idxs)
        
        val_size, te_size = int(all_size * valid_ratio), int(all_size * test_ratio)
        idx_val = idxs[:val_size]
        idx_te = idxs[val_size:val_size + te_size]
        idx_tr = idxs[val_size + te_size:]
        
        df_tr = self.df.iloc[idx_tr,:].copy()
        df_tr.reset_index(drop=True, inplace=True)
        df_val = self.df.iloc[idx_val,:].copy()
        df_val.reset_index(drop=True, inplace=True)
        df_te = self.df.iloc[idx_te,:].copy()
        df_te.reset_index(drop=True, inplace=True)
        
        return SequenceDataset(df_tr, self.vectorizer_dict, self.state),\
                SequenceDataset(df_te, self.vectorizer_dict, self.state),\
                SequenceDataset(df_val, self.vectorizer_dict, self.state)
    
    def sample(self, sample_num=None, sample_ratio=None, random_state=0):
        
        all_size = self.length
        idxs = np.arange(all_size)
        random.seed(random_state)
        random.shuffle(idxs)
        
        if sample_num is not None:
            idx_sample = idxs[:sample_num]
        elif sample_ratio is not None:
            idx_sample = idxs[:int(all_size * sample_ratio)]
        else:
            return None
        df_sample = self.df.iloc[idx_sample,:].copy()
        df_sample.reset_index(drop=True, inplace=True)
        return SequenceDataset(df_sample, self.vectorizer_dict, self.state)


    def __over_sampling(self, y, ratio=1.0):
        """
        Resample the dataset
        """
        label = np.unique(y)
        stats_c_ = {}
        maj_n = 0
        for i in label:
            nk = sum(y==i)
            stats_c_[i] = nk
            if nk > maj_n:
                maj_n = nk 
                maj_c_ = i

        idx = np.arange(len(y))
        # Keep the samples from the majority class
        idx_oversampled = idx[y == maj_c_]

        # Loop over the other classes over picking at random
        for key in stats_c_.keys():

            # If this is the majority class, skip it
            if key == maj_c_:
                continue

            # Define the number of sample to create
            num_samples = int(stats_c_[maj_c_] * ratio - stats_c_[key])

            # Pick some elements at random
            random_state = check_random_state(42)
            indx = random_state.randint(low = 0, high = stats_c_[key], size = num_samples)

            # Concatenate to the majority class
            idx_oversampled = np.concatenate([idx_oversampled, idx[y == key], idx[y == key][indx]])
        return idx_oversampled

