# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Made Marshall Vira Deva
# Udayana University
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95 and logs.get('val_accuracy')>0.95):
            print("\nHentikan Training ketika sudah memenuhi target.")
            self.model.stop_training = True


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    sentence = []
    labels = []

    for index, row in bbc.iterrows():
        labels.append(row[0])
        sentence.append(row[1])

    training_size = int(len(sentence) * training_portion)

    training_sentences =sentence[:training_size]
    training_labels = labels[:training_size]
    validation_sentences = sentence[training_size:]
    validation_labels = labels[training_size:]#YOUR CODE HERE

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index # YOUR CODE HERE

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded_sequences = pad_sequences(training_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded_sequences = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length,truncating=trunc_type)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    label_word_index = label_tokenizer.word_index
    training_label_sequences = label_tokenizer.texts_to_sequences(training_labels)
    training_label_sequences = np.array(training_label_sequences)
    validation_label_sequences = label_tokenizer.texts_to_sequences(validation_labels)
    validation_label_sequences = np.array(validation_label_sequences)
    # You can also use Tokenizer to encode your label.

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(
        training_padded_sequences,
        training_label_sequences,
        epochs=200,
        callbacks = CustomCallback(),
        validation_data=(
            validation_padded_sequences,
            validation_label_sequences),
        verbose=2)
    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
