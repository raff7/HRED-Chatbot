from lib.Constants import con
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize
def retrieve_line(line_ids, file_lines):
    output = []

    for id in line_ids:
        output.append(file_lines.loc[id][0])

    return output
def load_word2vec_Data_DISABLED():
        conversation_list = load_training_Data()
       # sentences = []
       # for conversation in conversation_list:
       #     for sentence in conversation:
       #         sentences.append(sentence)
        #return sentences
        return    [sentence for conv in conversation_list for sentence in conv]
def load_word2vec_Data():
        path = con.word2vec_train_file
        with open(path+'dialogues_text.txt', encoding="utf8") as file:
            sentences = []
            for line in file:
                for sentence in line.rstrip().split("__eou__"):
                    s = [x.lower() for x in re.findall(r"[\w']+|[.,!?;:]",str(sentence)) if x not in ['']]
                    if(len(s)>0):
                        s.insert(0,con.begin_of_sentence)
                        s.insert(len(s),con.end_of_sentence)
                        sentences.append(s)
                #return the single sentences to train w2v model
        return  sentences

def load_training_Data_DISABLED():
    
    path = con.word2vec_train_file
    file_lines = pd.read_csv(path + 'movie_lines.txt', sep=' \+\+\+\$\+\+\+ ', header=None, engine='python').iloc[:, [0, 4]]
    file_conversations = pd.read_csv(path + 'movie_conversations.txt', sep=' \+\+\+\$\+\+\+ ', header=None, engine='python').iloc[:, 3]

    file_conversations = file_conversations.apply(eval)
    file_lines.columns = ['id', 'line']

    file_lines = file_lines.set_index('id')

    conversation_list = file_conversations.apply(lambda x: retrieve_line(x, file_lines))
    conv_list = []
    for conversation in conversation_list:
        conv = []
        for sentence in conversation:
           sent = re.findall(r"[\w']+|[.,!?;:]",str(sentence))
           sent.insert(0,con.begin_of_sentence)
           sent.insert(len(sent),con.end_of_sentence)
           conv.append(sent)
        conv_list.append(conv)
    return conv_list

def load_training_Data():
    path = con.seq2seq_train_file
    with open(path+'dialogues_text.txt', encoding="utf8") as file:
        text_data = []
        st=[]
        for line in file:
            
            sent = []
            
            for sentence in line.rstrip().split("__eou__"):
                
                if(len(sentence)>0):
                    s = [x.lower() for x in re.findall(r"[\w']+|[.,!?;:]",str(sentence)) if x not in [''] and not any(char.isdigit() for char in x)]
                    if(len(s)<=con.max_len_sentence):
                        con.count +=1
                        s.insert(0,con.begin_of_sentence)
                        s.insert(len(s),con.end_of_sentence)
                        sent.append(s)
                        
                        st.extend(s)
                    else:
                        break
            if(len(sent)>1):
                text_data.append(sent)
    #return the text data divided hierarchically
    from collections import Counter
    counter = Counter(st)
    voc = counter.most_common(con.voc_size)
    voc =[x for x,i in voc]
    voc = np.insert(voc,0,con.pad)
    voc = np.insert(voc,1,con.unknown)
    return  text_data, voc
