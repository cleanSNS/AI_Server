import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding
from tensorflow.keras.optimizers import Adam

from basic_bert_model import BasicBertModel
from unklist import unklist

bert_model = BasicBertModel(unklist)

bert_layer = bert_model.get_bert_layer()
bert_tokenizer = bert_model.get_bert_tokenizer()
bert_tokenizer.add_tokens('!검열!')

class SwearWordDetector:
    def __init__(self):
        self.set_model()

    def set_model(self):
        input_ids = tf.keras.Input(shape=(100,),dtype='int32',name='input_ids')
        attention_masks = tf.keras.Input(shape=(100,),dtype='int32',name='attention_masks')

        output = bert_layer([input_ids,attention_masks],output_attentions = True)
        net = output['last_hidden_state']

        net = tf.keras.layers.Dense(4,activation='softmax')(net)
        outputs = net

        self.model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = outputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    def load_weights(self, path):
        self.model.load_weights(path)
        
    def summary(self):
        self.model.summary()
        
    def predict(self, input_ids, attention_masks):
        return self.model.predict([input_ids, attention_masks])
    

def bert_encode(data, max_len) :
    input_ids = []
    attention_masks = []

    for text in data:
      input_id = []
      attention_mask = []
      input_id.append(2)
      attention_mask.append(1)
      for word in text:
        if len(input_id) >= max_len -1:
          break
        encoded = bert_tokenizer.convert_tokens_to_ids(word)
        input_id.append(encoded)
        attention_mask.append(1)
      input_id.append(3)
      attention_mask.append(1)
      while len(input_id) <max_len:
        input_id.append(0)
        attention_mask.append(0)

      input_ids.append(input_id)
      attention_masks.append(attention_mask)


    return np.array(input_ids),np.array(attention_masks)

    
def bert_ner_encode(data, max_len) :
    input_ids = []

    for text in data:
      input_id = []

      input_id.append(0)
      for word in text:
        if len(input_id) >= max_len -1:
          break
        if word== 'O':
          input_id.append(1)
          continue
        if word == 'B':
          input_id.append(2)
          continue
        if word == 'I':
          input_id.append(3)
          continue
      input_id.append(0)
      while len(input_id) <max_len:
        input_id.append(0)

      input_ids.append(input_id)


    return np.array(input_ids)

