from transformers import TFBertModel
from transformers import AutoTokenizer
from unklist import unklist

class BasicBertModel:
    def __init__(self, unklist):
        self.download_model()
        self.init_tokenizer(unklist)
        
    def download_model(self):
        self.model = TFBertModel.from_pretrained("klue/bert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    def init_tokenizer(self,unklist):
        for word in unklist:
            self.tokenizer.add_tokens(word)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def get_bert_layer(self):
        return self.model
    
    def get_bert_tokenizer(self):
        return self.init_tokenizer
    

# bert_model = Basic_Bert_Model(unklist)

# bert_layer = bert_model.get_bert_layer()

# bert_tokenizer = bert_model.get_bert_tokenizer()