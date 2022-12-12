from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from swear_word_detector import SwearWordDetector, bert_encode, bert_tokenizer
from asdf import sequences_to_tag

swd = SwearWordDetector()

swd.summary()
swd.load_weights('002/hi')

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
    res = swd.predict(now_inputs, now_attentions)
    print(sequences_to_tag(res))
    print(now_inputs)
    return { "res" : coverup_swears(now_inputs[0], sequences_to_tag(res)[0]) }
