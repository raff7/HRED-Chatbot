
from nn import HRED
from lib.Constants import con
import run
from wordEmbeddings import word2vec as w2v
from wordEmbeddings import DataShaping as ds
import numpy as np
import re
from scipy import spatial

class chatbot:
    
    def __init__(self,wo2ve = None,encoder = None, decoder = None):
       # load
       self.hred = HRED()
       if(wo2ve is None):
           self.w2v = w2v()
       else:
           self.w2v = wo2ve
       self.ds = ds(self.w2v)
       if(encoder is None):
           self.encoder = self.hred.load_models(con.save_as_enc)
       else:
           self.encoder = encoder
       if(decoder is None):    
           self.decoder = self.hred.load_models(con.save_as_dec)
       else:
           self.decoder = decoder
       self.final_model = self.hred.build_final_model(self.encoder, self.decoder)
       self.final_model = self.hred.model_compile(self.final_model)
       

    
    def talk(self,input_sentence):
        input_sentence = input_sentence.lower()+" __EOS__"
        input_vector = []
        input_vector = self.ds.encoder_input_data_shaping(input_vector, re.findall(r"[\w']+|[.,!?;:]",str(input_sentence)))
        input_vector = np.array(input_vector)
        
        state_h, state_c = self.encoder.predict(input_vector)
        
        state_h = np.array(state_h)
        state_c = np.array(state_c)
    
        answer_sentence = []
        word = np.array([self.w2v.get_vec(con.begin_of_sentence)])
        word = np.reshape(word,(1,1,))
        stop_condition = False
        while not stop_condition:
            Nword_ = self.decoder.predict([word, state_h, state_c])
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

        state_h1 = np.array(state_h1)
        state_c1 = np.array(state_c1)
        state_h2 = np.array(state_h2)
        state_c2 = np.array(state_c2)
        
        word = np.array([self.w2v.get_vec(con.begin_of_sentence)])
        word = np.reshape(word,(1,1,))
 
        Nword1 = self.decoder.predict([word, state_h1, state_c1])
        Nword2 = self.decoder.predict([word, state_h2, state_c2])
       
        return spatial.distance.euclidean(Nword1, Nword2),1-spatial.distance.cosine(state_h1, state_h2),1-spatial.distance.cosine(state_c1, state_c2)



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
