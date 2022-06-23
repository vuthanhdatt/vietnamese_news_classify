# Vietnamese news classification with fine tuning PhoBERT pretrained model

***


## Demo

![Demo](https://github.com/vuthanhdatt/vietnamese_news_classify/blob/main/data/demo.gif?raw=true)
***

Run model in [colab](https://colab.research.google.com/drive/1Tu9Rwr_HRvQNWZaSWMBBuNEMZQ6905p1?usp=sharing)

## How to use
- Clone this repository to your local machine.
```bash
git clone https://github.com/vuthanhdatt/vietnamese_news_classify.git
```
- Create virtual environment.
```bash
python3 -m venv .your-virtual-name
```
- Activate virtual environment.
```bash
source .your-virtual-name/bin/activate
```
- Install dependencies.
```bash
pip install -r requirements.txt
```
- Run the code.
```bash
python3 app.py
```
- Go to locahost with port `5000` to see the result.

You can also interact with the result in `predict.ipynb` notebook.

## How to this project was created

This project fine tunning PhoBERT pretrained model to classify Vietnamese news and trained with over 42,000 titles in 5 categories from VnExpress.

### Data

Data was collected from 20th of June 2021 to 20th of June 2022. All the code to get data in `data\get_data.py`.
```
    Title	                                                Category
0	Bão Dianmu vào Thừa Thiên Huế - Quảng Ngãi	        Thời sự
1	Nadal bị đánh giá thấp hơn Djokovic và Alcaraz...	Thể thao
2	Cháy 10 cửa hàng ở Hà Nội	                        Thời sự
3	Belarus nói quân đội Ukraine 'bất mãn với Tổng...	Thế giới
4	Liverpool thua khi Salah hỏng phạt đền	            Thể thao
```
### Processing data

I use [underthesea](https://github.com/undertheseanlp/underthesea) for word segment and [PhoBERT](https://huggingface.co/PhoBERT) for tokenization.

```python
text = "Thủ tướng Nhật cảnh báo 'Đông Á có thể là Ukraine tiếp theo'"
preprocess(text)
'thủ_tướng nhật cảnh_báo đông_á có_thể là ukraine tiếp_theo'
```

```python
phobert = AutoModel.from_pretrained("vinai/phobert-base", return_dict = False)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", return_dict = False)

text = 'thủ_tướng nhật cảnh_báo đông_á có_thể là ukraine tiếp_theo'
tokenizer.(text)

{'input_ids': [0, 4739, 21697, 1223, 35352, 6992, 62, 8, 1881, 22055, 3403, 1512, 2], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

tokenizer.decode(tokenizer(text)['input_ids'])
<s> thủ_tướng nhật cảnh_báo đông_á có_thể là ukraine tiếp_theo </s>
```
### Dataloader

Create data loader for input data. See more on [pytorch docs](https://pytorch.org/docs/stable/data.html).

### Training

Trainning was instance with `Trainer` and `PhoBERT` custom class.

```python
model = PhoBert(
    phobert=phobert, dropout_p=dropout_p,
    embedding_dim=embedding_dim, num_classes=num_classes)
model = model.to(device)
trainer = Trainer(
    model=model, device=device, loss_fn=loss_fn, 
    optimizer=optimizer, scheduler=scheduler)
trainer.train(num_epochs, patience, train_dataloader, val_dataloader)

Epoch: 1 | train_loss: 0.00003, val_loss: 0.00002, lr: 1.00E-04, _patience: 10
Epoch: 2 | train_loss: 0.00002, val_loss: 0.00002, lr: 1.00E-04, _patience: 9
Epoch: 3 | train_loss: 0.00001, val_loss: 0.00002, lr: 1.00E-04, _patience: 10
Epoch: 4 | train_loss: 0.00001, val_loss: 0.00002, lr: 1.00E-04, _patience: 10
Epoch: 5 | train_loss: 0.00001, val_loss: 0.00002, lr: 1.00E-04, _patience: 9
Epoch: 6 | train_loss: 0.00000, val_loss: 0.00002, lr: 1.00E-04, _patience: 8
Epoch: 7 | train_loss: 0.00000, val_loss: 0.00002, lr: 1.00E-04, _patience: 7
Epoch: 8 | train_loss: 0.00000, val_loss: 0.00002, lr: 1.00E-04, _patience: 6
Epoch: 9 | train_loss: 0.00000, val_loss: 0.00002, lr: 1.00E-04, _patience: 5
Epoch: 10 | train_loss: 0.00000, val_loss: 0.00002, lr: 1.00E-05, _patience: 4
Epoch: 11 | train_loss: 0.00000, val_loss: 0.00002, lr: 1.00E-05, _patience: 3
Epoch: 12 | train_loss: 0.00000, val_loss: 0.00002, lr: 1.00E-05, _patience: 2
Epoch: 13 | train_loss: 0.00000, val_loss: 0.00002, lr: 1.00E-05, _patience: 1
Stopping early!

```

### Evaluation

The evaluation result store in `model\performance.json` file

Overall, the model is able to classify the data with precision of around 0.92%. Thời sự category has the lowest precision, around 84%

```
{
  "overall": {
    "precision": 0.9254264897691822,
    "recall": 0.924386814560225,
    "f1": 0.9246180762437886,
    "num_samples": 6401.0
  },
  "class": {
    "Kinh doanh": {
      "precision": 0.9292035398230089,
      "recall": 0.9264705882352942,
      "f1": 0.9278350515463918,
      "num_samples": 1360.0
    },
    "Sức khỏe": {
      "precision": 0.9145850120870266,
      "recall": 0.8937007874015748,
      "f1": 0.9040223018717642,
      "num_samples": 1270.0
    },
    "Thế giới": {
      "precision": 0.9380222841225627,
      "recall": 0.9064602960969045,
      "f1": 0.9219712525667351,
      "num_samples": 1486.0
    },
    "Thể thao": {
      "precision": 0.9801071155317521,
      "recall": 0.9831158864159631,
      "f1": 0.9816091954022989,
      "num_samples": 1303.0
    },
    "Thời sự": {
      "precision": 0.8426013195098964,
      "recall": 0.9103869653767821,
      "f1": 0.8751835535976504,
      "num_samples": 982.0
    }
  }
}
```
### Saving model
 The model saved in `model\model.pt`, loading this model for prediction later.


### Deployment

This part is not revelant to the course, so I will not going to detail here.

## Acknowledgement

- [Made with ML](https://madewithml.com/)

- [Full stack deep learning](https://fullstackdeeplearning.com/spring2021/lecture-4/) 

- Pham Dinh Khanh [blog](https://phamdinhkhanh.github.io/2020/06/04/PhoBERT_Fairseq.html)





