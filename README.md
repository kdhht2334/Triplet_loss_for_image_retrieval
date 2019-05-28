# 🏖 Metric loss for image retrieval


💡 This repository is where you can experiment with image retrieval examples based on various datasets.


🔑 Currently, we used only representative __triplet loss__, but we'll update various metric losses in the future.


## Dependencies

`tensorflow>=1.10.0`

`keras>=2.2.0`

## Usage

1. `pip install tensorflow-gpu==1.13.0`

2. `pip install keras`

3. `git clone https://github.com/kdhht2334/Triplet_loss_for_image_retrieval`

4. `cd Triplet_loss_for_image_retrieval/`

5. `python src/train_fashion_mnist.py`

6. or check `/notebook/train_fashion_mnist.ipynb`

## Description

To build model, simply do like below.

```python
gen = TripletGenerator()
train_stream = gen.flow(x_train, y_train, batch_size=batch_size)
valid_stream = gen.flow(x_valid, y_valid, batch_size=batch_size)

t = TripletNet(shape=input_size, dimensions=embedding_dimensions,
                               pretrained=False, learning_rate=0.001)
t.summary()
```

And to train, just use `fit_generator` in Keras API.
```python
for i in range(5):
    t.model.fit_generator(
            train_stream, 2500, epochs=1, verbose=1,
            callbacks=[checkpoint],
            validation_data=valid_stream, validation_steps=20)
```

And I also implemented source code about `Recall@K`.
```python
test_recall_at_one = np.mean(recall_at_kappa_support_query(x_supp_emb, y_supp, 
                                                       x_valid_emb, y_valid, 
                                                       kappa=kappa, dist=dist))

print("[INFO] Recall@{} is {}".format(kappa, test_recall_at_one))
                                                      
```
                                                     

Finally, you can use `find_l2_distance` to calculate the relationship between data samples based on the metric distance (Here we use l2 distance).

```python
if dist == 'l2':
    distances = find_l2_distances(x_valid_emb, x_supp_emb)
```


## Results

This is the results of image retrieval based on fashion-MNIST.

<p align="center">
  <img width="820" height="200" src="/pic/retrieval_result_1.png">
</p>

<p align="center">
  <img width="820" height="200" src="/pic/retrieval_result_2.png">
</p>

And we can also check embedding space of trained network using visualization tools.

<p align="center">
  <img width="820" height="200" src="/pic/visualization_of_embedding_space.png">
</p>


## Reference

- Triplet loss paper [[Paper]](https://arxiv.org/abs/1503.03832)

- Survey of metric learning [[Link]](https://github.com/kdhht2334/Survey_of_Deep_Metric_Learning)


## MileStone

- [x] Upload MNIST, fashion-MNIST basic examples
- [ ] Add another metric loss
- [ ] Add another dataset example