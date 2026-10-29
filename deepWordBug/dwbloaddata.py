import sys
import csv
csv.field_size_limit(2147483647)
from deepWordBug.preprocessing import Tokenizer, pad_sequences
import torch

# small = True
small = False
if small:
    textdatafolder = 'textdata_small/'
else:
    textdatafolder = 'textdata/'
    

            
class dataset:
    def __init__(self,filename,line=3):
        self.output = []
        self.content = []
        self.columns=line
        self.loadcsv(filename)
        
    def loadcsv(self, filename):
        reader = csv.reader(open(filename, "rt", encoding = "utf8"))
        count = 0
        for row in reader:
            if not row:
                continue
            if self.columns==2:
                self.output.append(int(row[0])-1)
                self.content.append((row[1]).lower())
            elif self.columns==3:
                self.output.append(int(row[0])-1)
                self.content.append((row[1] + " " + row[2]).lower())           
            elif self.columns==4:
                self.output.append(int(row[0])-1)
                self.content.append((row[1] + " " + row[2] + " " + row[3]).lower())       
def loaddata(i = 0):
    datanames = ['ag_news','amazon_review_full','amazon_review_polarity','dbpedia','sogou_news','yahoo_answers','yelp_review_full','yelp_review_polarity','enron','blog-authorship-corpus']
    lines = [3,3,3,3,3,4,2,2,2,2]
    classes= [4,5,2,14,5,10,5,2,2,3]
    # if not 'blog' in datanames[i]:
    trainadd = textdatafolder+datanames[i]+'_csv/train.csv'
    testadd = textdatafolder+datanames[i]+'_csv/test.csv'
    traindata = dataset(trainadd,lines[i])
    testdata = dataset(testadd,lines[i])
    return (traindata,testdata,classes[i])
    # else:
    #     data = dataset()
    #     return (traindata,testdata,classes[i])

def loaddatawithtokenize(testdata, nb_words, datalen, start_char = 1, oov_char=2, index_from=3):
    rawtest = testdata[:]
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(testdata)
    testdata = tokenizer.texts_to_sequences(testdata)
    if start_char==None:
        testdata = [[w + index_from for w in x] for x in testdata]
    else:
        testdata = [[start_char]+[w + index_from for w in x] for x in testdata]
    testdata = [[w if w < nb_words else oov_char for w in x] for x in testdata]
    testdata = pad_sequences(testdata, maxlen=datalen)
    return testdata,tokenizer,rawtest

def custom_collate_fn(batch):
    inputs = []
    targets = []
    idxs = []
    raws = []

    for item in batch:
        inputs.append(item[0])  # Tensor data (e.g., word indices)
        targets.append(item[1]) # Labels/targets
        idxs.append(item[2])    # Indexes
        raws.append(item[3])    # Raw text data

    # Convert lists to tensors or appropriate format
    inputs = torch.stack(inputs)  # Assuming inputs need to be stacked into a tensor
    targets = torch.tensor(targets)  # Assuming targets can be converted to a tensor
    return inputs, targets, idxs, raws
    
if __name__ == "__main__":
    (traindata,testdata,numclass) = loaddata(9)
    print(len(traindata.output))
    # blogdata('textdata/blog-authorship-corpus_csv/blogtext.csv',2)