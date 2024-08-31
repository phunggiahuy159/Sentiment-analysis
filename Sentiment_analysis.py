from datasets import load_dataset
from transformers import AdamW, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch 
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score

imdb=load_dataset('imdb')
raw_train, raw_test=imdb['train'],imdb['test']
'''{'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['neg', 'pos'], id=None)}'''
checkpoint="bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentence=tokenizer(imdb["train"]["text"])

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized_datasets=imdb.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader=DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
test_dataloader=DataLoader(tokenized_datasets["test"], batch_size=8, collate_fn=data_collator)


glove_file_path=r'D:\code\nlp\test\glove.6B.300d.txt'#link to glove file dim=300
embedding_dim=300
word_to_vec={}
with open(glove_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        values=line.split()
        word=values[0]
        vector=np.asarray(values[1:], dtype='float32')
        word_to_vec[word]=vector
# Create a vocabulary and embedding matrix
vocab=tokenizer.get_vocab()
vocab_size=len(vocab)
E=np.zeros((vocab_size, embedding_dim))
'''the index of the word in vocab must be the same with the embedding matrix'''
for word,idx in vocab.items():
    if word in word_to_vec:
        E[idx]=word_to_vec[word]
    else:
        E[idx]=np.random.normal(scale=0.6, size=(embedding_dim,))    
embedding_matrix=torch.tensor(E, dtype=torch.float32)
embedding_layer=nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
class TextModel(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.embedding=embedding_layer 
        self.lstm=nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc=nn.Linear(hidden_dim*2, 2)  
    def forward(self, x):
        embedded=self.embedding(x)
        _, (hidden, _)=self.lstm(embedded)
        hidden=torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output=self.fc(hidden)
        return output
    
hidden_dim=256
num_layers=2
lr=3e-4
epochs=4

model=TextModel(hidden_dim, num_layers)
optimizer=torch.optim.Adam(model.parameters(), lr=lr)
criterion=nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()  
    total_loss=0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
        input_ids=batch["input_ids"]
        labels=batch["labels"]
        outputs=model(input_ids)
        loss=criterion(outputs, labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    avg_loss=total_loss/len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")


model.eval()
true_labels=[]
predicted_labels=[]

with torch.no_grad():
    for batch in test_dataloader:
        input_ids=batch["input_ids"]
        labels=batch["labels"]

        outputs=model(input_ids)
        _, preds=torch.max(outputs, dim=1)

        true_labels.extend(labels.numpy())
        predicted_labels.extend(preds.numpy())

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Test Accuracy: {accuracy}")

torch.save(model.state_dict(), "text_classification_model.pth")

# Some random sentences
def preprocess_sentence(sentence):
    inputs=tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512, padding=True)
    return inputs["input_ids"]
def predict_sentiment(sentence):
    model.eval()
    input_ids=preprocess_sentence(sentence)
    
    with torch.no_grad():
        output=model(input_ids)
        _, predicted_label=torch.max(output, dim=1)
    
    return "Positive" if predicted_label.item()==1 else "Negative"

random_sentence="I enjoy the movie! It was a fantastic experience"
print(f"Sentence: '{random_sentence}' Sentiment: {predict_sentiment(random_sentence)}")

random_sentence="I didn't enjoy the movie at all. The ending is quite boring "
print(f"Sentence: '{random_sentence}' Sentiment: {predict_sentiment(random_sentence)}")
'''Test Accuracy: 0.88796'''
