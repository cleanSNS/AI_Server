from transformers import TFBertModel
from transformers import AutoTokenizer

bert_layer = TFBertModel.from_pretrained('klue/bert-base', from_pt=True)
bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

unklist = ['좆', '봊', '먀', '빸', '챂', '숩', '늗', '땃', '듸', '첬', '앍', '놐', 'ㅘ', '쌔', '듐', '쉑', '칰', '삘', '쥑', '찟', '됀', '깽', '놧', '췬', '쥴', '혓', '쉔', '랰', '켯', '읆', '밬', '횐', '걌', '빱', '넝', '겈', '싴', '뎃', '튱', '곗', '긔', '찎', '벆', '쨩', '줫', '퇘', '뭍', '잫', '끍', '듁', '켕', '뎀', '밮', '좉', '밈', '젼', '뺴', '랖', '겻', '맀', '뀝', '좡', '븍', '좨', '벍', '깄', '윀', '쳣', '딲', '뵌', '떄', '즇', '겜', '쌩', '좃', '띵', '풉', '넒', '햅', 'ㄸ', '뜀', '갘', '뭑', '쐬', 'ㅉ', '뭬', '쨈', '줴', '켚', '뀜', '섮', '쫒', '떵', 'ㅍ', '욀', '꿎', '뒈', '줜', '갣', '뮨', '튄', '젋', '얔', '믈', '홬', '톼', '싄', '빰', '튭', '냬', '쏩', '섿', '젴', '솎', '꼐', '떔', 'ㅊ', '챨', '쉈', '쪗', '퍠', '돠', '삥', '쫭', '먄', '곸', '쫑', 'ㄲ', '젆', '뼉', '꼰', '짞', '쎘', '띨', '얜', '맠', '괎', '퉷', '짴', '핬', '쎼', '꿘', '썪', '쒝', '썅', '됬', '쨔', '쎠', '샠', '헸', '줒', '잋', '녜', '얭', '촤', '챠', '졋', '뻇', '웜', '뇸', '뺌', '웟', '팤', '븡', '넜', '됫', '빐', '듦', '뿟', '겅', '튈', '귐', '줮', 'ㅄ', '맄', '셍', '욧', '봌', 'ㅖ', '귝', '킄', '쭐', '옘', 'ㅔ', '냨', '뻉', '샙', '퍄', '밪', '죈', '썌', '츈', '뜼', '콬', '텆', 'ㅑ', '겡', '갬', '셌', '짘', '픗', 'ㅗ', '몀', '깤', '뷸', '괸', '굼', '잨', '뵹', '헙', '텤', '냒', '긿', '팹', '룐', '쳌', '낑', '쿰', '뗴', '찝', '떰', '쒸', '씸', '뱁', '밠', '깰', '놋', '킵', '슼', '넼', '욲', '팼', 'ㅃ', '쥄', '쮸', '쌋', '왤', '넙', '뎈', '됭', '헴', '삔', '웩', '컽', '뽄', '웡', '쌘', '뗀', 'ㅛ', '짰', '됌', '갗', '얍', '퀀', '묜', '뷴', '듕', '쎾', '쨰', '훠', '윸', '쟜', '눼', '줏', '뽁', '롴', '됏', '젔', '탰', '텄', '씝', '붭', '뻬', '뿎', '탉', '삯', '츌', '쑈', '앜', '휼', '룟', '셧', '떢', '텼', '핥', '솝', '닼', '귈', '옜', '볏', '읎', 'ㅆ', '휑', '찣', '앂', '눜', '똔', '홧', '귭', '쫍', '밡', '갉', '쳒', '숴', '힛', '뼛', '돸', '펩', '몃', '뺄', '퉤', '믜', '봣', '짢', '벱', '핻', '뀨', '쉣', 'ㅐ', '잌', '꽈', '셴', '훅', '죸', '덛', '뇽', '벵', '깠', '뇬', '죶', '캇', '깟', '긱', '엡', '핰', '졵', '댜', '렠', '챗', '껀', '솤', '뿅', '읔', '쳡', '샄', '걘', '앎', '빳', '슙', '쯉', '쥡', '믁', '횽', '듄', '녓', '잰', '앴', '긑', '쇅', '샥', '잽', '콱', '죤', '붊', '쐇', '낔', '얏', '뢀', '꺠', '씐', '븨', '졍', '껒', '펨', '쥰', '봁', '줵', '떽', '띡', '돨', '얖', '샾', '붇', '븃', '짆', '댘', '옶', '엊', '읏', '윾', '곘', '죗', '짹', '럲', '꺄', '꼇', '쨋', '빻', '햐', '쟨', '랔', '캍', '밯', '꿏', '겤', '묭', '쯧', '슝', 'ㅓ', '똣', '빕', '쩝', '텓', '횃', '룜', '싀', '읺', '겞', '걑', '겋', '쭵', '핳', '컹', '쎆', '뻡', '떙', '닠', '뭥', 'ㄳ', '겆', '핼', '냘', '죵', '쐈', '꽐', '뗐', '돤', '뤗', '쉨', '븅', '셩', '갭', '놬', '짯', '졷', '깉', '컄', '궜', '썻', '줭', '쎔', '킁', '뱄', '앖', '퀵', '힂', '쎈', '쭝', '핌', '낰', '쌉', '쑵', '딨', 'ㅕ', '쳥', '끠', '틍', '츙', '껬', '헹', '껸', '넵', '쥿', '됴', '놰', '앝', '좍', '깼', '꾠', '꽹', '퓹', '휜', '첧', '쩃', '섺', '핔', '폇', '괕', '밷', '껰', '졔', '땟', '깈', '젭', '퍅', '뤈', '헠', '헀', 'ㄻ', '봬', '첰', '쌰', '쀜', '빢', '읜', '듥', '둣', '궸', '쪾', '챡', '믐', '럐', '뭠', '짼', '쉿', '퀰', '긎', '됑', '웁', '궉', '핟', '퐈', '넫', '맊', '딫', '냔', '짩', '캥', '죙', '괱', '뫠', '츤', '즁', '깸', 'ㅙ', '댱', '뫈', '껜', '햝', '놤', '좠', '혬', '쓲', '켐', '햬', '짛', '긇', '쉡', '셕', '밐', '쎇', '젹', '읫', '믓', '돚', '튠', '읶', '뮺', '넽', '줳', '읋', '쿹', '샜', '얶', '밞', '쉒', '퓻', '녗', '샸', '춌', '돜', '윳', '읇', '죧', '픕', '쨀', '홋', '핮', '쟘', '쒯', '꿉', '앷', '됔', '괬', '뗌', '쏨', '왝', '굔', '뽚', '좸', '떈', '붂', '떌', '뽈', '쒀', '갮', '쟎', '켴', '맜', '펰', '맆', '꿧', '벸', '멷', '홰', '댸', '좽', '옙', '욌', '섻', '퍞', '짇', '즤', '틔', '톽', '똨', '쩧', '랒', '귯', '뗜', '쿳', '앢', '삑', '갰', '큔', '돛', '똬', '욏', '솽', '쪙', '퀼', '멓', '틑', '멱', '픠', '깞', '쎗', '볐', '똠', '쫘', '쩨', '랊', '쎅', '뷱', '뎁', '롸', '꼌', '뚷', '똭', '쿱', '밗', '뺸', '릏', '샊', '왱', '횬', '땔', '씪', '섔', '넚', '켲', '쏫', '씰', '쎌', '킺', '챤', '귱', '껼', '켬', '왑', '읃', '뉸', '꺀', '뻨', '옇', '띃', '퉅', '녈', '꽊', '넴', '맅', '췤', '듫', '읻', '튐', '좈', '젗', '츨', '굄', '잴', '킽', '큄', '뱌', '찤', '쥽', '봥', '엲', '맼', '섴', '졉', '샨', '뎟', '햔', '쟝', '쭙', '췹', '뼤']

for word in unklist:
  bert_tokenizer.add_tokens(word)

bert_layer.resize_token_embeddings(len(bert_tokenizer))


import re
import pandas as pd
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

embedding_dim = 128
hidden_units = 256

input_ids = tf.keras.Input(shape=(100,),dtype='int32',name='input_ids')
attention_masks = tf.keras.Input(shape=(100,),dtype='int32',name='attention_masks')

output = bert_layer([input_ids,attention_masks],output_attentions = True)
net = output['last_hidden_state']

#net = tf.keras.layers.Dense(128,activation='relu')(net)
#net = tf.keras.layers.Dropout(0.2)(net)

#net = tf.keras.layers.Dense(8,activation='relu')(net)
#net = tf.keras.layers.Dropout(0.2)(net)
net = tf.keras.layers.Dense(4,activation='softmax')(net)
outputs = net

model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


model.load_weights('002/hi')



import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('./DataWordPiece.csv')
data = data.iloc[:39343,1:]

data['Word'] = data['Word'].str.lower()
print('Word 열의 중복을 제거한 값의 개수 : {}'.format(data.Word.nunique()))

func = lambda temp: [(w, t) for w, t in zip(temp["Word"].values.tolist(), temp["Tag"].values.tolist())]
tagged_sentences=[t for t in data.groupby("Sentence #").apply(func)]
print("전체 샘플 개수: {}".format(len(tagged_sentences)))

sentences, ner_tags = [], []
for tagged_sentence in tagged_sentences: # 47,959개의 문장 샘플을 1개씩 불러온다.

    # 각 샘플에서 단어들은 sentence에 개체명 태깅 정보들은 tag_info에 저장.
    sentence, tag_info = zip(*tagged_sentence)
    sentences.append(list(sentence)) # 각 샘플에서 단어 정보만 저장한다.
    ner_tags.append(list(tag_info)) # 각 샘플에서 개체명 태깅 정보만 저장한다

tar_tokenizer = Tokenizer(lower=False)

tar_tokenizer.fit_on_texts(ner_tags)


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

train_input_ids, train_attention_masks = bert_encode(sentences,100)

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

y_data = bert_ner_encode(ner_tags,100)

ner_to_index = tar_tokenizer.word_index
index_to_ner = tar_tokenizer.index_word
index_to_ner[0] = 'PAD'

print(index_to_ner)

train_input_ids, test_input_ids, train_attention_masks,test_attention_masks, y_train_int, y_test_int = train_test_split(train_input_ids,train_attention_masks, y_data, test_size=.2, random_state=777)

y_train = to_categorical(y_train_int, num_classes=4)
y_test = to_categorical(y_test_int, num_classes=4)



from seqeval.metrics import f1_score, classification_report

def sequences_to_tag(sequences):
    result = []
    # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
    for sequence in sequences:
        word_sequence = []
        # 시퀀스로부터 확률 벡터 또는 원-핫 벡터를 하나씩 꺼낸다.
        for pred in sequence:
            # 정수로 변환. 예를 들어 pred가 [0, 0, 1, 0 ,0]라면 1의 인덱스인 2를 리턴한다.
            pred_index = np.argmax(pred)
            # index_to_ner을 사용하여 정수를 태깅 정보로 변환. 'PAD'는 'O'로 변경.
            word_sequence.append(index_to_ner[pred_index].replace("PAD", "O"))
        result.append(word_sequence)
    return result

#y_predicted = model.predict([test_input_ids, test_attention_masks])
#pred_tags = sequences_to_tag(y_predicted)
#test_tags = sequences_to_tag(y_test)


#print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))
#print(classification_report(test_tags, pred_tags))

bert_tokenizer.add_tokens('!검열!')


from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class Item(BaseModel):
    sentence: str

def coverup_swears(encoded, predict):
    #print(encoded)
    #print(predict)
    res = []
    for i in range(1,len(encoded)):
        if encoded[i] == 3:
            return bert_tokenizer.decode(res)
        elif encoded[i] == 1 or encoded[i] == 32000 or predict[i] == 'B' or predict[i] == 'I':
            res.append(32707)
        else:
            res.append(encoded[i])
    return bert_tokenizer.decode(res)


@app.get("/")
def read():
    print('hi')
    return {"Hello":"World"}

@app.post("/")
def read_root(item: Item):
    #print('hi')
    print(item)
    now_inputs, now_attentions = bert_encode([bert_tokenizer.tokenize(item.sentence)], 100)
    res = model.predict([now_inputs, now_attentions])
    print(sequences_to_tag(res))
    print(now_inputs)
    return { "res" : coverup_swears(now_inputs[0], sequences_to_tag(res)[0]) }
