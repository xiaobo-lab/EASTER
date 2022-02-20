import numpy as np
import tensorflow as tf
from abc import ABC
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification

import tools


MODEL_PATH = r"cardiffnlp/twitter-roberta-base-sentiment"


class DataGenerator(ABC, Sequence):
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH)

    def __init__(self, texts, labels):
        from sklearn.preprocessing import OneHotEncoder

        max_len = 0
        input_ids = []
        attention_mask = []
        encoder = OneHotEncoder().fit([[-1], [0], [1]])
        for text in texts:
            t = DataGenerator.tokenizer(text, return_tensors='tf')
            input_ids.append(t['input_ids'].numpy())
            attention_mask.append(t['attention_mask'].numpy())
            max_len = max(len(input_ids[-1][0]), max_len)
        self.max_len = max_len
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = encoder.transform(np.asarray(labels).reshape((-1, 1))).toarray()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, batch_ids):
        return {'input_ids': self.input_ids[batch_ids], 'attention_mask': self.attention_mask[batch_ids]}, \
               np.array([self.labels[batch_ids]])

    def pad_to(self, length):
        self.input_ids = [i.reshape((1, -1)) for i in
                          tools.pad_sequences([i[0] for i in self.input_ids], length, 'zero')]
        self.attention_mask = [i.reshape((1, -1)) for i in
                               tools.pad_sequences([i[0] for i in self.attention_mask], length, 'same')]
        self.max_len = max(self.max_len, length)
        return self


def build_model(kernel_size, filters, strides, units):
    # load roberta model
    roberta = TFRobertaForSequenceClassification.from_pretrained(MODEL_PATH)

    # define model inputs
    inputs = {'input_ids': layers.Input((None,), dtype=tf.int32),
              'attention_mask': layers.Input((None,), dtype=tf.int32)}

    roberta_main = roberta.layers[0](inputs)[0]

    # one road
    conv_outputs = []
    for size, filter_, stride in zip(kernel_size, filters, strides):
        x = layers.Conv1D(filter_, size, stride, activation='relu')(roberta_main)
        x = layers.GlobalMaxPool1D()(x)
        conv_outputs.append(x)
    x = layers.concatenate(conv_outputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(units, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_0 = layers.Dense(3, activation='relu')(x)

    # another road
    output_1 = roberta.layers[1](roberta_main)

    x = layers.concatenate([output_0, output_1])
    outputs = layers.Dense(3, activation='softmax')(x)
    model_ = Model(inputs, outputs)
    model_.compile(optimizers.Adam(5e-6), 'categorical_crossentropy', ['accuracy'])
    return model_


def load_model():
    model = build_model(kernel_size=(2, 2, 3), filters=(300, 300, 200), strides=(2, 1, 1), units=150)
    model.load_weights('./data/EASTER.h5')
    return model


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = list(i.rsplit(';', 2) for i in f.read().split('\n')[1:] if len(i) > 3)
    return tuple(map(lambda x: x[0], data)), tuple(map(lambda x: int(x[1]), data))


def load_4423(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = list(i.split(';', 2) for i in f.read().split('\n') if len(i) > 3)
    return tuple(map(lambda x: x[2], data)), tuple(map(lambda x: {'positive': 1, 'negative': -1, 'neutral': 0}[x[1]], data))


def evaluate(model, path):
    texts, labels = load_data(path)
    data = DataGenerator(texts, labels)
    pred_y = model.predict(data)
    pred_y = np.where(pred_y == np.max(pred_y, axis=1).reshape(-1, 1))[1]
    real_y = np.asarray(labels) + 1
    res = tools.cal_quota(real_y, pred_y, 0, 1, 2, True)


def evaluate4423(model, path):
    texts, labels = load_4423(path)
    data = DataGenerator(texts, labels)
    pred_y = model.predict(data)
    pred_y = np.where(pred_y == np.max(pred_y, axis=1).reshape(-1, 1))[1]
    real_y = np.asarray(labels) + 1
    res = tools.cal_quota(real_y, pred_y, 0, 1, 2, True)


def plot_model(model):
    from tensorflow.keras.utils import plot_model

    plot_model(model, to_file='./data/model.png', show_shapes=True, show_layer_names=False)


def train():
    train_texts, train_labels = load_4423('./data/train3097.csv')
    test_texts, test_labels = load_4423('./data/test1326.csv')
    train_data = DataGenerator(train_texts, train_labels).pad_to(3)
    test_data = DataGenerator(test_texts, test_labels).pad_to(3)
    callbacks = [ModelCheckpoint('./data/EASTER.h5', 'val_accuracy', 1, True, True),
                 EarlyStopping(patience=10)]
    model = build_model(kernel_size=(2, 2, 3), filters=(300, 300, 200), strides=(2, 1, 1), units=150)
    model.fit(train_data, epochs=50, validation_data=test_data, callbacks=callbacks, verbose=1)


def test():
    model = load_model()
    print('all:sof4423')
    evaluate(model, './data/sof4423.csv')
    print('all:sof1500')
    evaluate(model, './data/sof1500.csv')
    print('all:app_review')
    evaluate(model, './data/app_review.csv')
    print('all:jira')
    evaluate(model, './data/jira.csv')
    print('4423:')
    evaluate4423(model, './data/test1326.csv')


if __name__ == "__main__":
    # train()
    test()
