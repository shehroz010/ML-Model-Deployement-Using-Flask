import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert

import pandas as pd
import re
import numpy as np
import random, math
import os

movie_reviews = pd.read_csv("C:/Users/shehr/Desktop/DataSets/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

movie_reviews.isnull().values.any()
movie_reviews.shape


def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return RE.sub('', text)

reviews = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    reviews.append(preprocess_text(sen))


y = movie_reviews['sentiment']
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))


BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocab_file, lower_case)


def tokenize_reviews(text_reviews):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))


tokenized_reviews = [tokenize_reviews(review) for review in reviews]


reviews_with_len = [ [review, y[i], len(review)] for i, review in enumerate(tokenized_reviews) ]


random.shuffle(reviews_with_len)
reviews_with_len.sort(key=lambda x: x[2])


sorted_reviews_labels = [(review_lab[0], review_lab[1]) for review_lab in reviews_with_len]


final_dataset = tf.data.Dataset.from_generator(lambda: sorted_reviews_labels, output_types=(tf.int32, tf.int32))


BATCH_SIZE = 32
batched_dataset = final_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))


TOTAL_BATCHES = math.ceil(len(sorted_reviews_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 10
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)


class TEXT_MODEL(tf.keras.Model):
    
    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()
        
        self.dense_1 = layers.Dense(units = dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate = dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units = 1,
                                           activation = "sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation = "softmax")
    
    def call(self, inputs, training):
        l = self.embedding(inputs)
        l1 = self.cnn_layer1(l) 
        l1 = self.pool(l1) 
        l2 = self.cnn_layer2(l) 
        l2 = self.pool(l2)
        l3 = self.cnn_layer3(l)
        l3 = self.pool(l3) 
        
        concatenated = tf.concat([l1, l2, l3], axis=-1)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)
        
        return model_output


VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 2
DROPOUT_RATE = 0.2
NB_EPOCHS = 10


text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)


checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = 'C:/Users/shehr/BERT.hdf5', verbose = 1, save_best_only = True )


if OUTPUT_CLASSES == 2:
    text_model.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
else:
    text_model.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])


text_model.fit(train_data, validation_data = test_data, epochs=10, callbacks = [checkpointer])


#results = text_model.evaluate(test_data)
#print(results)
