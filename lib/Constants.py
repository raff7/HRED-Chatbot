
class con():
    ''' valiable setting'''
    embedding_size = 100
    encoder_output_size= 300
    batch_size = 1
    
    enc_dropout = 0.4
    enc_r_dropout = 0.2
    dec_dropout = 0.4
    dec_r_dropout = 0.2
    
    depth = 1
    epochs = 1
    datapoints = 80
    iterations = 15
    
    max_answer=25
    max_len_sentence = 25
    save_every = 10 *60
    print1 = False
    print_iterations = False
    ''' fata settings '''
    begin_of_sentence = "__BOS__"
    end_of_sentence = "__EOS__"
    pad = "__PAD__"
    '''directory setting'''
    project_dir = "C:/Users/User/Documents/DKE/Thesis/HRED_GitHub"
    save_as_enc =  'model_encoder.hdf5'#DS2
    save_as_dec =  'model_decoder.hdf5'
    loss = 'loss.csv'
    dloss = 'dloss.csv'
    ''' word2vec '''
    word2vec_train_file = project_dir + "/lib/Data/"
    word2vec_model = project_dir + '/lib/models/w2vModel'
    voc_path = project_dir + '/lib/models/vocabulary'

    ''' seq2seq '''
    seq2seq_models_save_dir = project_dir + '/lib/models/'
    seq2seq_train_file = project_dir + "/lib/Data/"


    always_best = True
        
    count = 0