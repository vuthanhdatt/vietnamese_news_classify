import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import json
import re
from transformers import AutoModel, AutoTokenizer
from underthesea import word_tokenize

from flask import Flask, request
app = Flask(__name__)

class Predict(object):
  def __init__(self, model, device):

        # Set params
        self.model = model
        self.device = device
  def predict(self, dataloader):
        """Prediction step."""
        # Set model to eval mode
        self.model.eval()
        y_probs = []

        # Iterate over val batches
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):

                # Forward pass w/ inputs
                inputs, targets = batch[:-1], batch[-1]
                z = self.model(inputs)

                # Store outputs
                y_prob = F.softmax(z).cpu().numpy()
                y_probs.extend(y_prob)

        return np.vstack(y_probs)

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, ids, masks, targets):
        self.ids = ids
        self.masks = masks
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index):
        ids = torch.tensor(self.ids[index], dtype=torch.long)
        masks = torch.tensor(self.masks[index], dtype=torch.long)
        targets = torch.FloatTensor(self.targets[index])
        return ids, masks, targets

    def create_dataloader(self, batch_size, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=False)
def preprocess(text):
    """Conditional preprocessing on our text unique to our task."""
    # Lower
    text = text.lower()

    # Remove words in paranthesis
    text = re.sub(r'\([^)]*\)', '', text)

    # Spacing and filters
    text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
    text = re.sub(r'[^A-Za-z0-9wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+',' ',text)# remove non alphanumeric chars
    text = re.sub(' +', ' ', text)  # remove multiple spaces
    text = text.strip()
   
    return  word_tokenize(text, format='text')

class LabelEncoder(object):
    """Label encoder for tag labels."""
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def encode(self, y):
        y_one_hot = np.zeros((len(y), len(self.class_to_index)), dtype=int)
        for i, item in enumerate(y):
            y_one_hot[i][self.class_to_index[item]] = 1
        return y_one_hot

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            index = np.where(item == 1)[0][0]
            classes.append(self.index_to_class[index])
        return classes

    @classmethod
    def load(cls, fp):
        with open(fp, "r", encoding='utf-8') as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)

class PhoBert(nn.Module):
    def __init__(self, phobert, dropout_p, embedding_dim, num_classes):
        super(PhoBert, self).__init__()
        self.phobert = phobert
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc1 = torch.nn.Linear(embedding_dim, num_classes)
    
    def forward(self, inputs):
        ids, masks = inputs
        seq, pool = self.phobert(input_ids=ids, attention_mask=masks)
        z = self.dropout(pool)
        z = self.fc1(z)
        return z

def get_probability_distribution(y_prob, classes):
    """Create a dict of class probabilities from an array."""
    results = {}
    for i, class_ in enumerate(classes):
        results[class_] = np.float64(y_prob[i])
    sorted_results = {k: v for k, v in sorted(
        results.items(), key=lambda item: item[1], reverse=True)}
    return sorted_results

@app.route('/predict', methods=['POST','GET'])
def predict_label():
    global predict_model
    global tokenizer
    global le
    if request.method == 'POST':
        text = request.form['text']
        X = preprocess(text)
        encoded_input = tokenizer(X, return_tensors="pt", padding=True).to(torch.device("cpu"))
        ids = encoded_input["input_ids"]
        masks = encoded_input["attention_mask"]
        y_filler = le.encode([le.classes[0]]*len(ids))
        dataset = DataLoader(ids=ids, masks=masks, targets=y_filler)
        dataloader = dataset.create_dataloader(batch_size=128)
        y_prob = predict_model.predict(dataloader)
        y_pred = np.argmax(y_prob, axis=1)
        result= le.index_to_class[y_pred[0]]
        prob_dist = get_probability_distribution(y_prob=y_prob[0], classes=le.classes)
        return f'''
        <form method="POST">
               <textarea name="text" rows="4" cols="50">{text}</textarea>
               <br>
               <input type="submit" value="Submit">
                <div>Category: <b>{result}</b></div>
                <br>
                <div><b>Probability:</b></div>
                <div>Thời sự: {prob_dist['Thời sự']}</div>
                <div>Thế giới: {prob_dist['Thế giới']}</div>
                <div>Kinh doanh: {prob_dist['Kinh doanh']}</div>
                <div>Thể thao: {prob_dist['Thể thao']}</div>
                <div>Sức khỏe: {prob_dist['Sức khỏe']}</div>

        </form>
        '''
    return '''
           <form method="POST">
                <textarea name="text" rows="4" cols="50"></textarea>
               <br>
               <input type="submit" value="Submit">
           </form>'''

if __name__ == '__main__':
    le = LabelEncoder().load("model/label_encoder.json")

    device = torch.device("cpu")
    dropout_p = 0.5
    phobert = AutoModel.from_pretrained("vinai/phobert-base", return_dict = False)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", return_dict = False)
    embedding_dim = phobert.config.hidden_size
    model = PhoBert(
    phobert=phobert, dropout_p=dropout_p,
    embedding_dim=embedding_dim, num_classes=len(le))

    model.load_state_dict(torch.load("model/model.pt", map_location=torch.device('cpu')))
    model.to(device)

    predict_model = Predict(model=model, device=device)

    app.run(host='0.0.0.0', port=5000)