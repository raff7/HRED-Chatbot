
class con():
    ''' valiable setting'''
    embedding_size = 100
    encoder_output_size= 300
    batch_size = 100
    
    enc_dropout = 0.5
    enc_r_dropout = 0.2
    dec_dropout = 0.5
    dec_r_dropout = 0.2
    
    depth = 0
    epochs = 1
    datapoints = 15 *2
    iterations = 1
    
    voc_size = 5000
    validation = 0.2
    max_answer=25
    max_len_sentence = 25
    save_every = 0.1 *60
    print1 = False
    print_iterations = True
    
    Load_Model = True
    ''' fata settings '''
    begin_of_sentence = "__BOS__"
    end_of_sentence = "__EOS__"
    pad = "__PAD__"
    unknown = '__UNK__'
    '''directory setting'''
    project_dir = "C:/Users/User/Documents/DKE/Thesis/HRED_GitHub"
    save_as_enc =  'model_encoder-d'+str(depth)+' ['+str(embedding_size)+','+str(encoder_output_size)+'] '+str(batch_size)+' '+str(datapoints/2)+' '+str(iterations)+'.hdf5'#DS2
    save_as_dec =  'model_decoder-d'+str(depth)+' ['+str(embedding_size)+','+str(encoder_output_size)+'] '+str(batch_size)+' '+str(datapoints/2)+' '+str(iterations)+'.hdf5'
    save_as_con =  'model_context-d'+str(depth)+' ['+str(embedding_size)+','+str(encoder_output_size)+'] '+str(batch_size)+' '+str(datapoints/2)+' '+str(iterations)+'.hdf5'
    loss = 'loss-d'+str(depth)+' ['+str(embedding_size)+','+str(encoder_output_size)+'] '+str(batch_size)+' '+str(datapoints/2)+' '+str(iterations)+'.csv'
    dloss = 'dloss-d'+str(depth)+' ['+str(embedding_size)+','+str(encoder_output_size)+'] '+str(batch_size)+' '+str(datapoints/2)+' '+str(iterations)+'.csv'
    val_loss = 'val_loss-d'+str(depth)+' ['+str(embedding_size)+','+str(encoder_output_size)+'] '+str(batch_size)+' '+str(datapoints/2)+' '+str(iterations)+'.csv'
    val_dloss = 'val_dloss-d'+str(depth)+' ['+str(embedding_size)+','+str(encoder_output_size)+'] '+str(batch_size)+' '+str(datapoints/2)+' '+str(iterations)+'.csv'
    ''' word2vec '''
    word2vec_train_file = project_dir + "/lib/Data/"
    word2vec_model = project_dir + '/lib/models/w2vModel'
    voc_path = project_dir + '/lib/models/vocabulary'

    ''' seq2seq '''
    seq2seq_models_save_dir = project_dir + '/lib/models/'
    seq2seq_train_file = project_dir + "/lib/Data/"


    always_best = True
        
    count = 0