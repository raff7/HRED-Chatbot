
from nn import HRED
from lib.Constants import con
import run
from wordEmbeddings import word2vec as w2v
from wordEmbeddings import DataShaping as ds
import numpy as np
import re
from scipy import spatial

class chatbot:
    
    def __init__(self,wo2ve = None,encoder = None, decoder = None,context = None):
       # load
        if(wo2ve is None):
           self.w2v = w2v()
        else:
           self.w2v = wo2ve
        self.hred = HRED(self.w2v)
        self.ds = ds(self.w2v)
        if(encoder is None):
           self.encoder = self.hred.load_models(con.save_as_enc)
        else:
           self.encoder = encoder
        if(decoder is None):    
           self.decoder = self.hred.load_models(con.save_as_dec)
        else:
           self.decoder = decoder
        if(context is None):    
           self.context = self.hred.load_models(con.save_as_con)
        else:
           self.context = context
        
        self.init_hidden_states()
        self.final_model = self.hred.build_final_model(self.encoder, self.decoder,self.context)
        self.final_model = self.hred.model_compile(self.final_model)
       

    
    def talk(self,input_sentence):
        input_sentence = input_sentence.lower()+" __EOS__"
        input_vector = []
        input_vector = self.ds.encoder_input_data_shaping(input_vector, re.findall(r"[\w']+|[.,!?;:]",str(input_sentence)))
        input_vector = np.array(input_vector)
        
        state_h, state_c = self.encoder.predict(input_vector)
        
        state_h = state_h.reshape(1, 1, con.encoder_output_size)

        _, self.meta_hh, self.meta_hc = self.context.predict([state_h, self.meta_hh, self.meta_hc])
    
        answer_sentence = []
        word = np.array([self.w2v.get_vec(con.begin_of_sentence)])
        word = np.reshape(word,(1,1,))
        stop_condition = False
        while not stop_condition:
            Nword_ = self.decoder.predict([word, self.meta_hh, self.meta_hc])
            Nword = self.w2v.get_word_prob(Nword_)
            word =  np.array([np.append(word[0],[Nword],axis=0)])
            #word=Nword
            answer_sentence.append(Nword)
            if (self.w2v.get_word(Nword) ==con.end_of_sentence or len(answer_sentence) >= con.max_answer):
                stop_condition = True
        return self.w2v.vec_to_sentence(answer_sentence)
    
    def check(self):
        input_vector1 = []
        input_vector2 = []
        input_sentence1="I'm exhausted.".lower()+"  __EOS__"
        input_sentence2='The kitchen stinks.'.lower()+" __EOS__"
        input_vector1 = self.ds.encoder_input_data_shaping(input_vector1, re.findall(r"[\w']+|[.,!?;:]",str(input_sentence1)))
        input_vector2 = self.ds.encoder_input_data_shaping(input_vector2, re.findall(r"[\w']+|[.,!?;:]",str(input_sentence2)))

        input_vector1 = np.array(input_vector1)
        input_vector2 = np.array(input_vector2)

        state_h1, state_c1 = self.encoder.predict(input_vector1)
        state_h2, state_c2 = self.encoder.predict(input_vector2)

        state_h1 = state_h1.reshape(1, 1, con.encoder_output_size)
        state_h2 = state_h2.reshape(1, 1, con.encoder_output_size)
        self.init_hidden_states()
        _, meta_hh1, meta_hc1 = self.context.predict([state_h1, self.meta_hh , self.meta_hc])
        _, meta_hh2, meta_hc2 = self.context.predict([state_h2, self.meta_hh, self.meta_hc])
        
        word = np.array([self.w2v.get_vec(con.begin_of_sentence)])
        word = np.reshape(word,(1,1,))
 
        Nword1 = self.decoder.predict([word, meta_hh1, meta_hc1])
        Nword2 = self.decoder.predict([word, meta_hh1, meta_hc2])
       
        return spatial.distance.euclidean(Nword1, Nword2),1-spatial.distance.cosine(meta_hh1, meta_hh2),1-spatial.distance.cosine(meta_hc1, meta_hc2)
    
    def init_hidden_states(self):
        self.meta_hh = np.array([[0]*con.encoder_output_size])
        self.meta_hc = np.array([[0]*con.encoder_output_size])

if __name__ == '__main__':           
        print('Chatting...')
        bot = chatbot()
        bot.check()
        bot.talk('going to the gym is healthy')
        bot.talk('what are you talking about?')
        question = input('Hi! write something\n')

        context_state = None

        while question.lower().find('exit') == -1:
            if question.lower().find('reset') == -1:
                answer = bot.talk(question)
                print('A) -> ' + answer)
                question = input() 
            else:
                bot.resetContext()
