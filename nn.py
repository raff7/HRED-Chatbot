from keras.models import Sequential, Model
from keras.layers import Input, LSTM, GRU, RNN, Dense,Bidirectional, Reshape, Embedding
import keras
import pydot
from keras.utils import plot_model
from keras import backend as K
from lib.Constants import con
import wordEmbeddings
import datetime
from keras.callbacks import LambdaCallback


class HRED():
    def __init__(self):
        self.input_size = con.embedding_size
        self.latent_dim = con.encoder_output_size
        self.w2v=wordEmbeddings.word2vec()
    def build_encoder(self):
        K.set_learning_phase(1) # set learning phase

        encoder_inputs = Input(shape=(None,))
        LSTM_input = Embedding(input_dim=len(self.w2v.voc), output_dim=self.input_size, weights=[self.w2v.get_embedding_matrix()],mask_zero=True, trainable = True)(encoder_inputs)
        for i in range(con.depth):
            LSTM_input = LSTM(self.latent_dim, return_sequences=True , dropout=con.enc_dropout, recurrent_dropout=con.enc_r_dropout)(LSTM_input)
        _, state_h, state_c = LSTM(self.latent_dim, return_state=True, dropout=con.enc_dropout, recurrent_dropout=con.enc_r_dropout)(LSTM_input)

        return Model(encoder_inputs, [state_h, state_c])


    def build_decoder(self):
        K.set_learning_phase(1) # set learning phase

        encoder_h = Input(shape=(self.latent_dim,))
        encoder_c = Input(shape=(self.latent_dim,))
        encoder_states = [encoder_h, encoder_c]

        decoder_inputs = Input(shape=(None,))
        LSTM_input = Embedding(input_dim=len(self.w2v.voc), output_dim=self.input_size, weights=[self.w2v.get_embedding_matrix()],mask_zero=True, trainable = True)(decoder_inputs)
        for i in range(con.depth):
            LSTM_input = LSTM(self.latent_dim, return_sequences=True, dropout=con.dec_dropout, recurrent_dropout=con.dec_r_dropout)(LSTM_input)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=False, return_state=False)
        decoder_outputs= decoder_lstm(LSTM_input, initial_state=encoder_states)
        decoder_outputs = Dense(self.input_size, activation='relu')(decoder_outputs)
        decoder_outputs = Dense(len(self.w2v.voc), activation='softmax')(decoder_outputs)

        return Model([decoder_inputs, encoder_h, encoder_c], decoder_outputs)



    def build_final_model(self, encoder, decoder):
        # encoder
        encoder_inputs = Input(shape=(None,))
        layer = encoder.layers
        LSTM_input = layer[1](encoder_inputs)
        for i in range(con.depth):    
            LSTM_input = layer[i+2](LSTM_input)
        encoder_output, state_h, state_c = layer[len(layer)-1](LSTM_input)
        encoder_states = [state_h, state_c]

        # decoder
        decoder_inputs = Input(shape=(None,))
        layer2 = decoder.layers
        LSTM_input = layer2[1](decoder_inputs)
        for i in range(con.depth):
            LSTM_input = layer2[i+2](LSTM_input)
        decoder_lstm_outputs =  layer2[con.depth+4](LSTM_input, initial_state=encoder_states)
        decoder_outputs =  layer2[con.depth+5](decoder_lstm_outputs)
        outputs =  layer2[con.depth+6](decoder_outputs)

        return Model([encoder_inputs, decoder_inputs], outputs)


    def model_compile(self, model):
        #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        optimizer = 'rmsprop'
        loss = 'categorical_crossentropy'
        model.compile(optimizer=optimizer,loss=loss, metrics=['accuracy'])
        #model.summary()
        return model

    def train_final_model(self, model, encoder_input_data, decoder_input_data, decoder_target_data):
        ver= False
        if(con.print1):#(datetime.datetime.now() < datetime.datetime(2018,5,31,10,23)) or(datetime.datetime.now() > datetime.datetime(2018,6,30,8)) ):
            ver=True
        if(ver):
            print("Q: %s A: %s W: %s"%(self.w2v.vec_to_sentence(encoder_input_data[0]), self.w2v.vec_to_sentence(decoder_input_data[0]),self.w2v.vec_to_sentence([i for i,x in enumerate(decoder_target_data[0]) if x == 1])))
        #print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[con.depth+3].get_weights()))
        loss = model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=len(encoder_input_data), epochs=con.epochs,verbose=ver)#,callbacks = [print_weights])#,validation_split=0.2)
        return loss




    def save_models(self, fname, model):
        print("save "+con.seq2seq_models_save_dir+fname)
        model.save(con.seq2seq_models_save_dir+fname)

    def load_models(self, fname):
        print("load "+con.seq2seq_models_save_dir+fname)
        from keras.models import load_model
        return load_model(con.seq2seq_models_save_dir+fname)