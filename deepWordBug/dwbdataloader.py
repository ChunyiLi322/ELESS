import torch
from torch.utils.data import DataLoader, Dataset

class Worddata(Dataset):
    def __init__(self, data, tokenizer = True, length=1014, space = False, backward = -1, getidx = False, rawdata = None):
        self.backward = backward
        self.length = length
        (self.inputs,self.labels) = (data['content'],data['output'])
        self.labels = torch.LongTensor(self.labels)
        self.inputs = torch.from_numpy(self.inputs).long()
        self.getidx = getidx
        if rawdata:
            self.raw = rawdata
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self,idx):
        x = self.inputs[idx]
        y = self.labels[idx]
        if self.getidx==True:
            if self.raw:
                return x,y,idx,self.raw[idx]
            else:
                return x,y,idx
        else:
            return x,y
if __name__ == '__main__':
    # Example for generating external dataset
    import pickle
    import loaddata
    (train,test,tokenizer,numclass) = loaddata.loaddatawithtokenize(0)
    test.content = test.content[:100]
    test.output = test.output[:100]
    word_index = tokenizer.word_index
    
    pickle.dump((test,word_index,numclass), open('textdata/ag_news_small_word.pickle','wb'))