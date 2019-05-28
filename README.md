# 🏖 Triplet loss for image retrieval


💡 This repository is where you can experiment with image retrieval examples based on various datasets.


🔑 Currently, we used only representative __triplet loss__, but we'll update various metric losses in the future.


## Dependencies

`tensorflow>=1.10.0`

`keras>=2.2.0`

## Usage

`1. pip install tensorflow-gpu==1.13.0`

`2. git clone https://github.com/kdhht2334/Triplet_loss_for_image_retrieval`

`3. cd Triplet_loss_for_image_retrieval/`

`4. python src/train_fashion_mnist.py`

## Description

To build model, simple like below
```python
gen = TripletGenerator()
train_stream = gen.flow(x_train, y_train, batch_size=batch_size)
valid_stream = gen.flow(x_valid, y_valid, batch_size=batch_size)

t = TripletNet(shape=input_size, dimensions=embedding_dimensions,
                               pretrained=False, learning_rate=0.001)
t.summary()
```

And to train, just use `fit_generator` in Keras API
```python
for i in range(5):
    t.model.fit_generator(
            train_stream, 2500, epochs=1, verbose=1,
            callbacks=[checkpoint],
            validation_data=valid_stream, validation_steps=20)
```

And I also implemented source code about recall@K
```python
test_recall_at_one = np.mean(recall_at_kappa_support_query(x_supp_emb, y_supp, 
                                                       x_valid_emb, y_valid, 
                                                       kappa=kappa, dist=dist))

print("[INFO] Recall@{} is {}".format(kappa, test_recall_at_one))
                                                      
                                                     ```
                                                     

Finally, you can use `find_l2_distance` to calculate the relationship between data samples based on the metric distance. (Here we use l2 distance)

```python
if dist == 'l2':
    distances = find_l2_distances(x_valid_emb, x_supp_emb)
```


## Results

This is the results of image retrieval

<p align="center">
  <img width="820" height="250" src="/pic/retrieval_result_1.png">
</p>

<p align="center">
  <img width="820" height="250" src="/pic/retrieval_result_2.png">
</p>
