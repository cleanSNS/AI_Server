import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from swear_word_detector import bert_encode, bert_ner_encode

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

train_input_ids, train_attention_masks = bert_encode(sentences,100)

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

# y_predicted = model.predict([test_input_ids, test_attention_masks])
# pred_tags = sequences_to_tag(y_predicted)
# test_tags = sequences_to_tag(y_test)


# print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))
# print(classification_report(test_tags, pred_tags))

