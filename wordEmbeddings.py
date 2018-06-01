#import nltk
#from nltk.corpus import inaugural
import gensim
import numpy as np
from lib.Constants import con
import lib.data as data
import random
class word2vec:
    def __init__(self,load=True, voc = None):
        if(load):
            print("Load Word2Vec")
            self.loadModel()
        else:
            print("MAKE NEW Word2Vec -------------- OVERWRITE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if(voc is None):
                print("ERROR, no vocabulary input.")
            self.trainModel(voc)
            
    def trainModel(self,voc):
        dt = data.load_word2vec_Data()
        model = gensim.models.Word2Vec(dt, size=con.embedding_size, min_count=1)#Train model
        #model = gensim.models.KeyedVectors.load("C:\\Users\\User\\Documents\\DKE\\Thesis\\model-subtitles")
        #pretrained model
        self.model = model
        self.voc = voc
        
        np.save(con.voc_path,self.voc)
        model.save(con.word2vec_model)
        return model
    def loadModel(self):
        #model = gensim.models.KeyedVectors.load(con.word2vec_model)
        self.voc = np.load(con.voc_path+'.npy')
        #self.model =model
        #return self.model
    def getEmb_data(self,text_data):
        emb_data = []
        for conversation in text_data:
            conv=[]
            for sentence in conversation:
                sent = []
                for word in sentence:
                    sent.append(self.get_vec(word))
                conv.append(sent)
            emb_data.append(conv)        
        return emb_data
    
    def get_word(self,i):
        return self.voc[i]
    
    def get_vec(self,word):
        if(word == con.pad):
            return 0
        if(word in self.voc):
            return [i for i, w in enumerate(self.voc) if w == word][0]
        else:   
            print('{0} is an out of dictionary word'.format(word))
            return 0
    
    def sentence_to_vec(self,sentence):
        vec = []
        for w in sentence:
            vec.append(self.get_vec(w))
        return vec 
    def string_to_vec(self,sentence):
        vec = []
        for w in sentence.split(" "):
            vec.append(self.get_vec(w))
        return vec 
    def vec_to_sentence(self,vec):
        sent = ''
        for i in vec:
            sent = sent+' '+(self.get_word(i))
        return sent
    def get_word_prob(self, probability_vector):
        #INPUT: probability vector
        #OUTPUT: Index chosen according to probabilities
        if(con.always_best):
            return probability_vector[0].tolist().index(max(probability_vector[0].tolist()))
        else:
            return np.random.choice(range(0,len(self.voc)),p=probability_vector[0]/probability_vector[0].sum(axis=0,keepdims=1))
    def get_embedding_matrix(self):
        
        embedding_weights = np.zeros((len(self.voc), con.embedding_size))
        for index,word in enumerate(self.voc):
            if(index > 0):
                embedding_weights[index, :] = self.model[word]
        return embedding_weights
class DataShaping:
    def __init__(self,w2v):
        self.w2v = w2v
            
    def make_data_seq(self,conversation,i):
        encoder_input_batch = []
        decoder_input_batch = []
        decoder_output_batch = []
       
        
        conv = conversation[::]
        #eoncoder input
        encoder_batch=[]
        for b in range(con.batch_size):
            encoder_batch.append(conv[b][i])
        encoder_batch = self.pad_sentences(encoder_batch)
        for b in range(con.batch_size):
            encoder_input_batch = self.encoder_input_data_shaping(encoder_input_batch, encoder_batch[b])

        #decoder inputs
        decoder_batch=[]
        for b in range(con.batch_size):
            decoder_batch.append(conv[b][i+1])
        decoder_batch = self.pad_sentences(decoder_batch)
        dec_batch_copy = decoder_batch[::]
        for b in range(con.batch_size):
            decoder_input_batch = self.decoder_input_data_shaping(decoder_input_batch, decoder_batch[b])
            decoder_output_batch = self.decoder_output_data_shaping(decoder_output_batch, dec_batch_copy[b])

        encoder_input_batch = np.array(encoder_input_batch)
        decoder_input_batch = np.array(decoder_input_batch)
        decoder_output_batch = np.array(decoder_output_batch)

        #print("train shape:", encoder_input_batch.shape)
        #print("teach shape:", decoder_input_batch.shape)
        #print("target shape:", decoder_output_batch.shape)

        return encoder_input_batch, decoder_input_batch, decoder_output_batch
        
    def add_BOS(self, s):
        sentence = s[::]
        if (con.begin_of_sentence not in sentence):
            sentence.insert(0, con.begin_of_sentence)
        return sentence

    def rm_BOS(self, s):
        sentence = s[::]
        if (con.begin_of_sentence in sentence):
            sentence.remove(con.begin_of_sentence)
        return sentence
    def rm_EOS(self, s):
        sentence = s[::]
        if (con.end_of_sentence in sentence):
            sentence.remove(con.end_of_sentence)
        return sentence
    
    def encoder_input_data_shaping(self,encoder_input_batch,enc_sentence):
        enc_sentence = self.rm_BOS(enc_sentence)
        #enc_sentence = enc_sentence[::-1]
        encoder_input_batch.append(self.w2v.sentence_to_vec(enc_sentence))
        return encoder_input_batch
    def decoder_input_data_shaping(self,decoder_input_batch,dec_sentence):
        dec_sentence = self.rm_EOS(dec_sentence)
        decoder_input_batch.append(self.w2v.sentence_to_vec(dec_sentence))
        return decoder_input_batch
    def decoder_output_data_shaping(self,decoder_output_batch,dec_sentence):
        dec_target_sentence = self.rm_BOS(dec_sentence)
        decoder_output_batch.append(self.w2v.sentence_to_vec(dec_target_sentence))
        return decoder_output_batch
    def pad_sentences(self,sentences):
        max_length = len(max(sentences,key=len))
        for s in sentences:
            if(len(s)<max_length):
                for i in range(max_length-len(s)):
                    s.append(con.pad)
        return sentences    
    def pad_same_length(self,batch_data):
        max_length = len(max(batch_data,key=len))
        for dp in batch_data:
            if(len(dp)<max_length):
                for i in range(max_length-len(dp)):
                    dp.append([con.begin_of_sentence,con.end_of_sentence])
        return batch_data, max_length
    def remove_extra_padding(self,data):
        for i in range(len(data[0])):
            j = len(data[0])-i
            for d in data:
                if(any(d[j-1] != [0]*con.embedding_size)):
                    return data[:,:j]
            
        
    def filter_padding(self, k, train_data, teach_data, target_data):#TBD WHY IS THE ENCODER PADDED OUT, IT SHOULD ONLY BE THE DECODER...
        step_train_data= []
        step_teach_data= []
        step_target_data = []
        for i in range(len(train_data)):
            if(not (target_data[i][k] == 0)):
                step_train_data.append(train_data[i])
                step_teach_data.append(teach_data[i,:k+1])
                step_target_data.append(self.to_one_hot(target_data[i,k]))
        return self.remove_extra_padding(np.array(step_train_data)),np.array(step_teach_data),np.array(step_target_data)
    
    def to_one_hot(self,i):
        one_hot = [0]*len(self.w2v.voc)
        one_hot[i]=1
        return one_hot