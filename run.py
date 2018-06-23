
import numpy as np
from keras.utils import plot_model
import random
from time import time
from lib.Constants import con
import lib.data as data
from wordEmbeddings import word2vec, DataShaping
from nn import HRED
from lib.Constants import con
import win_unicode_console
win_unicode_console.enable()
import chatbot

def train(loadModel=True):
    #LoadModel True  -> Load a model and continue training
    #LoadModel False -> Create new model, begin training
    text_data, voc = data.load_training_Data()     
    w2v = word2vec(loadModel, voc)
    hred = HRED(w2v)
    ds = DataShaping(w2v)
    
    
    if(loadModel):    #load model
        print("LOADING MODEL")
        encoder_model = hred.load_models(con.save_as_enc)
        decoder_model = hred.load_models(con.save_as_dec)
        context = hred.load_models(con.save_as_con)

       
    else:       #new model
        print("MAKE NEW MODEL ------------ OVERWRITE!!!!!!!!!!!!!!!!!!!!!!!!")
        encoder_model = hred.build_encoder()
        decoder_model = hred.build_decoder()
        context = hred.build_context_model()


        encoder_model = hred.model_compile(encoder_model)
        decoder_model = hred.model_compile(decoder_model)
        context = hred.model_compile(context)

        
    final_model = hred.build_final_model(encoder_model, decoder_model,context)
    final_model = hred.model_compile(final_model)
    
    #Sort the dataset to reduce padding
    text_data = sorted(text_data, key=len, reverse=False)
    restart_point =   0
    loss_history = []
    loss_detailed_history = []
    val_loss_history = []
    val_loss_detailed_history = []
    last_saved = time()
    print("BEGIN TRAINING")
    count=0
    cycle_count = 1
    #Training cycle
    while(True):
        l = len(text_data[count])
        adj_datapoints = max(1,round(con.datapoints*l/l**2))
        if(float(count/len(text_data))>=restart_point):
            los_cycle = 0
            val_los_cycle=0
            cycle_time = time()
            print("Group number %i, Conversation length:%i  datapoints: %i"%(cycle_count,l,adj_datapoints))
            #repeat the training of this subset 'con.iterations' times
            for iteration in range(0,con.iterations):
                #initialize context hidden states
                meta_hh, meta_hc = init_context_states()
                los_iteration=0
                val_los_iteration=0
                if(con.print_iterations):
                    print("iteration %i"%(iteration))
                    tb = time()
                #Go trough all the conversations, with a step size equal to the batch size
                for j in range(0,adj_datapoints*con.batch_size,con.batch_size):
                    batch_data = text_data[j+count:j+count+con.batch_size]

                    if(len(batch_data)<1):
                        break
                    batch_data,max_length = ds.pad_same_length(batch_data)
                    los_i=0
                    val_los_i=0
                    #go trough sentences
                    for i in range(max_length-1):
                        #print()
                        los_k=0
                        val_los_k=0
                        train_data, teach_data, teach_target_data = ds.make_data_seq(batch_data, i) 
                        #print("Teach %i, Train %i"%(len(teach_data[0]),len(train_data[0])))
                        for k in range(len(teach_data[0])):
                            step_train_data, step_teach_data, step_target_data,step_meta_hh,step_meta_hc = ds.filter_padding(k,train_data,teach_data,teach_target_data,meta_hh,meta_hc)
                            
                            hist=hred.train_final_model(final_model, step_train_data, step_teach_data,step_target_data,step_meta_hh,step_meta_hc)#.history['loss']
                            
                            los=hist.history['loss']                         
                            val_los = hist.history['val_loss']
                            los_k += np.average(los)
                            val_los_k += np.average(val_los)
                        state_h, _ = encoder_model.predict(train_data)
                        state_h = state_h.reshape(con.batch_size, 1, con.encoder_output_size)
                        _, meta_hh, meta_hc = context.predict([state_h, meta_hh, meta_hc])    
                        los_k = los_k/len(teach_data[0])
                        los_i += los_k
                        val_los_k = val_los_k/len(teach_data[0])
                        val_los_i += val_los_k
                    los_i = los_i/(max_length-1)
                    los_iteration += los_i
                    val_los_i = val_los_i/(max_length-1)
                    val_los_iteration += val_los_i
                los_iteration = los_iteration/adj_datapoints
                loss_detailed_history.append(los_iteration)
                los_cycle += los_iteration
                val_los_iteration = val_los_iteration/adj_datapoints
                val_loss_detailed_history.append(val_los_iteration)
                val_los_cycle += val_los_iteration
                if(con.print_iterations):
                    print("iteration done in %f seconds, loss: %f,val_los: %f"%(time()-tb,los_iteration,val_los_iteration))
                    
            los_cycle = los_cycle/con.iterations
            loss_history.append(los_cycle)
            val_los_cycle = val_los_cycle/con.iterations
            val_loss_history.append(val_los_cycle)
            print("Cycle completed in %f seconds. Loss: %f, Val_Loss: %f"%(time()-cycle_time,los_cycle,val_los_cycle))
            if time() - last_saved >con.save_every:
                last_saved = time()
                hred.save_models(con.save_as_enc, encoder_model)
                hred.save_models(con.save_as_dec, decoder_model)
                hred.save_models(con.save_as_con, context)

                bot = chatbot.chatbot(encoder=encoder_model,decoder=decoder_model)
                cos_sim_out,cos_sim_h,cos_sim_c = bot.check()
    
                #test the bot on some inputs.   
                print("Saving, progress: %f"%( float(count/len(text_data))))
                print("Check different input, Distance Output: %f ,H: %f, C: %f"%(cos_sim_out,cos_sim_h,cos_sim_c))
                print("input1: Q) 'the kitchen stinks.' A) %s"%(bot.talk('the kitchen stinks.')))
                print("input2: Q) 'i'm exhausted.' A) %s"%(bot.talk("i'm exhausted.")))
                print("input1: Q) 'hi, how are you?' A) %s"%(bot.talk('hi, how are you?')))
                print("input2: Q) 'what do you think about America?' A) %s"%(bot.talk("what do you think about America?")))
                print("input1: Q) 'i don't believe what you said, you know?.' A) %s"%(bot.talk("i don't believe what you said, you know?")))
                print("input2: Q) 'go to hell!.' A) %s"%(bot.talk("go to hell!")))

                #save los
                np_loss_history = np.array(loss_history)
                np_loss_detailed_history = np.array(loss_detailed_history)
                np.savetxt(con.loss, np_loss_history)
                np.savetxt(con.dloss,np_loss_detailed_history)
                np_val_loss_history = np.array(val_loss_history)
                np_val_loss_detailed_history = np.array(val_loss_detailed_history)
                np.savetxt(con.val_loss, np_val_loss_history)
                np.savetxt(con.val_dloss,np_val_loss_detailed_history)
                    
        count = ((count+(adj_datapoints*con.batch_size)))
        if(count > len(text_data)):
            count = 0
        cycle_count += 1        
            
            
  
    
def init_context_states():
    #initialize hidden state
    meta_hh = np.array([[(0) for i in range(con.encoder_output_size)]for i in range(con.batch_size)])
    meta_hc = np.array([[(0) for i in range(con.encoder_output_size)]for i in range(con.batch_size)])
    return meta_hh, meta_hc              
  
          
if __name__ == '__main__': 
    train(loadModel = con.Load_Model)
