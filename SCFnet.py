'''
The code implementation of SCFnet under SynCF framework.
Essentially, this is based on the official implementation of CFNet under DeepCF framework.
'''

import numpy as np
import tensorflow as tf
from keras import initializers
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, concatenate, Dot, Lambda, multiply, Reshape
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras import backend as K
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import argparse
import PMF
import NCF

def parse_args():
    parser = argparse.ArgumentParser(description="Run SCFnet.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')    
    
    parser.add_argument('--commonuser',nargs = '?',default = '[256]',
                        help = 'User embedding layers shared by PMF and NCF')
    parser.add_argument('--commonitem',nargs = '?',default = '[256]',
                        help = 'Item embedding layers shared by PMF and NCF')
    parser.add_argument('--NCFlayers',nargs = '?',default = '[256,128,64]',
                        help = 'Size of each layers in the MLP part of NCF')
    parser.add_argument('--PMFlayers1',nargs = '?',default = '[64]',
                        help = 'Size of each user and item layer before outer product. '
                               'Notice we assume user and item layers have the same size')
    parser.add_argument('--PMFlayers2',nargs = '?',default = '[64]',
                        help = 'Size of each MLP prediction layer in PMF')

    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--mr', type=float, default=0.75,
                        help='Mixing rate. Weight of pretrained PMF parameters in initialized common layers')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='sgd',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--PMF_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for PMF part. If empty, no pretrain will be used')
    parser.add_argument('--NCF_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for NCF part. If empty, no pretrain will be used')
    return parser.parse_args()

def get_model(train, num_users, num_items, commonuser, commonitem, NCF1, PMF1, PMF2):
    common_num = len(commonuser)
    NCF_num = len(NCF1)
    PMF1_num = len(PMF1)
    PMF2_num = len(PMF2)
    
    # The whole training set is kept in model. Batches are taken by indexing it
    user_matrix = K.constant(getTrainMatrix(train))
    item_matrix = K.constant(getTrainMatrix(train).T)
    
    # Input variables whose values refer to a batch
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    
    user_rating= Lambda(lambda x: tf.gather(user_matrix, tf.to_int32(x)))(user_input)
    item_rating = Lambda(lambda x: tf.gather(item_matrix, tf.to_int32(x)))(item_input)
    user_rating = Reshape((num_items, ))(user_rating)
    item_rating = Reshape((num_users, ))(item_rating)
    print(user_rating.shape,item_rating.shape)
    
    # Common part of PMF and NCF
    userlayer = Dense(commonuser[0],  activation="linear" , name='user_common0')
    itemlayer = Dense(commonitem[0], activation="linear" , name='item_common0')
    common_user_latent = userlayer(user_rating)
    common_item_latent = itemlayer(item_rating)
    print(common_user_latent.shape,common_item_latent.shape)
    for idx in range(1,common_num):
        userlayer = Dense(commonuser[idx],  activation="relu" , name='user_common%d'%idx)
        itemlayer = Dense(commonitem[idx], activation="relu" , name='item_common%d'%idx)
        common_user_latent = userlayer(common_user_latent)
        common_item_latent = itemlayer(common_item_latent)        
        print(common_user_latent.shape,common_item_latent.shape)
    
    # MLP prediction part of NCF
    NCF_vector = concatenate([common_user_latent,common_item_latent])
    print(NCF_vector.shape)
    for idx in range(NCF_num):
        NCF_vector = Dense(NCF1[idx],activation = 'relu',name = 'NCF_layer%d'%idx)(NCF_vector)
        print(NCF_vector.shape)
        
    # PMF part1
    PMF_user = common_user_latent
    PMF_item = common_item_latent
    for idx in range(PMF1_num):
        PMF_user = Dense(PMF1[idx],activation = 'relu',name = 'PMF_user%d'%idx)(PMF_user)
        PMF_item = Dense(PMF1[idx],activation = 'relu',name = 'PMF_item%d'%idx)(PMF_item)
        print(PMF_user.shape,PMF_item.shape)
    
    # PMF part2. Tensor outer product is realized by matrix-column multiplication and concatenation
    _slice = Lambda(getslice,arguments = {'index':0})(PMF_item)
    PMF_vector = multiply([PMF_user,_slice])
    print(_slice.shape,PMF_vector.shape)
    for idx in range(PMF_item.shape[1]-1):
        _slice = Lambda(getslice,arguments = {'index':idx+1})(PMF_item)
        _slice = multiply([PMF_user,_slice])
        PMF_vector = concatenate([PMF_vector,_slice])
    print(PMF_vector.shape)

    # MLP prediction part of PMF 
    for idx in range(PMF2_num):
        PMF_vector = Dense(PMF2[idx],activation = 'relu',name = 'PMF_layer%d'%idx)(PMF_vector)
        print(PMF_vector.shape)

    # Concatenate DMF and MLP prediction parts
    predict_vector = concatenate([PMF_vector, NCF_vector])
    print(predict_vector.shape)
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer=initializers.lecun_normal(),
                       name="prediction")(predict_vector)
    
    model_ = Model(inputs=[user_input, item_input], outputs=prediction)
    
    return model_

# A util function for keras lambda layer in get_model
def getslice(x,index):
    return x[:,index]

# Convert a scipy sparse matrix dict to numpy matrix
def getTrainMatrix(train):
    num_users, num_items = train.shape
    train_matrix = np.zeros([num_users, num_items], dtype=np.int32)
    for (u, i) in train.keys():
        train_matrix[u][i] = 1
    return train_matrix

# Load pretrained PMF if required
def load_pretrain_PMF(model, PMF_model, common, PMF1, PMF2):
    common_num = len(common) # user/item_common0,1,...,common_num-1
    
    # Initialize common layer
    for idx in range(common_num):
        user_layer = PMF_model.get_layer('user_layer%d'%idx).get_weights()
        item_layer = PMF_model.get_layer('item_layer%d'%idx).get_weights()
        model.get_layer('user_common%d'%idx).set_weights(user_layer)
        model.get_layer('item_common%d'%idx).set_weights(item_layer)
    
    # Initialize PMF part1
    for idx in range(len(PMF1)):
        user_layer = PMF_model.get_layer('user_layer%d'%(idx+common_num)).get_weights()
        item_layer = PMF_model.get_layer('item_layer%d'%(idx+common_num)).get_weights()
        model.get_layer('PMF_user%d'%idx).set_weights(user_layer)
        model.get_layer('PMF_item%d'%idx).set_weights(item_layer)
    
    # Initialize PMF part2
    for idx in range(len(PMF2)):
        prediction_layer = PMF_model.get_layer('prediction%d'%idx).get_weights()
        model.get_layer('PMF_layer%d'%idx).set_weights(prediction_layer)
            
    # Initialize PMF prediction weights
    model_prediction = model.get_layer('prediction').get_weights()
    PMF_prediction = PMF_model.get_layer('predicion').get_weights()
    new_weights = np.concatenate((PMF_prediction[0],model_prediction[0][PMF2[-1]:]),axis = 0)
    new_b = PMF_prediction[1]
    model.get_layer('prediction').set_weights([new_weights, new_b]) 
    return model

# Load pretrained NCF if required
def load_pretrain_NCF(model, NCF_model, common, NCF1, mr):
    # Initialize common layer with mr:1-mr ratio between PMF and NCF 
    for idx in range(len(common)):
        user_layer = NCF_model.get_layer('user_layer%d'%idx).get_weights()
        item_layer = NCF_model.get_layer('item_layer%d'%idx).get_weights()
        user_common = model.get_layer('user_common%d'%idx).get_weights()
        item_common = model.get_layer('item_common%d'%idx).get_weights()
        new_user_weights = (1-mr)*user_layer[0]+mr*user_common[0]
        new_user_bias = (1-mr)*user_layer[1]+mr*user_common[1]
        new_item_weights = (1-mr)*item_layer[0]+mr*item_common[0]
        new_item_bias = (1-mr)*item_layer[1]+mr*item_common[1]
        model.get_layer('user_common%d'%idx).set_weights([new_user_weights,new_user_bias])
        model.get_layer('item_common%d'%idx).set_weights([new_item_weights,new_item_bias])

    # Initialize NCF prediction weights
    for idx in range(len(NCF1)):
        prediction_layer = NCF_model.get_layer('prediction%d'%idx).get_weights()
        model.get_layer('NCF_layer%d'%idx).set_weights(prediction_layer)
    
    # Initialize final prediction
    model_prediction = model.get_layer('prediction').get_weights()
    NCF_prediction = NCF_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((model_prediction[0][:-NCF1[-1]], NCF_prediction[0]), axis=0)
    new_b = model_prediction[1] + NCF_prediction[1]
    # 0.5 means the contributions of MF and MLP are equal
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b]) 
    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    commonuser = eval(args.commonuser)
    commonitem = eval(args.commonitem)
    NCFlayers = eval(args.NCFlayers)
    PMFlayers1 = eval(args.PMFlayers1)
    PMFlayers2 = eval(args.PMFlayers2)
    num_negatives = args.num_neg
    learner = args.learner
    mixing_rate = args.mr
    learning_rate = args.lr
    batch_size = args.batch_size
    num_epochs = args.epochs
    verbose = args.verbose
    PMF_pretrain = args.PMF_pretrain
    NCF_pretrain = args.NCF_pretrain
            
    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("SCFnet arguments: %s " % args)
    model_out_file = 'Pretrain/%s_SCFnet_%d.h5' %(args.dataset, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives   
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(train, num_users, num_items, commonuser, commonitem, NCFlayers, PMFlayers1, PMFlayers2)
    #print(model.summary())
    
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    
    # Load pretrain model
    if PMF_pretrain != '' and NCF_pretrain != '':
        PMF_model = PMF.get_model(train, num_users, num_items,
                                  commonuser+PMFlayers1, commonitem+PMFlayers1, PMFlayers2)
        PMF_model.load_weights(PMF_pretrain)
        model = load_pretrain_PMF(model,PMF_model,commonuser,PMFlayers1,PMFlayers2)
        del PMF_model
        NCF_model = NCF.get_model(train, num_users, num_items,
                                  commonuser, commonitem, NCFlayers)
        NCF_model.load_weights(NCF_pretrain)
        model = load_pretrain_NCF(model,NCF_model,commonuser,NCFlayers,mixing_rate)
        del NCF_model
        print("Load pretrained PMF (%s) and NCF (%s) models done. " % (PMF_pretrain, NCF_pretrain))
        
    # Check Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out > 0:
        model.save_weights(model_out_file, overwrite=True) 
        
    # Training model
    for epoch in range(num_epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0: 
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best SCFnet model is saved to %s" % model_out_file)
