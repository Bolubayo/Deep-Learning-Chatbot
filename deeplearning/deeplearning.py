import nltk
# nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import random
import json
import pickle

# Load data globally so available in chat()
with open("intents.json") as file:
    data = json.load(file)

try:
    with open('data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)
except:
    with open("intents.json") as file:
        data = json.load(file)

    # print(data)
    print(data["intents"])

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern, language='english')
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # training = numpy.array(training)
    # output = numpy.array(output)
    #
    # tensorflow.reset_default_graph()
    #
    # net = tflearn.input_data(shape=[None, len(training[0])])
    # net = tflearn.fully_connected(net, 8)
    # net = tflearn.fully_connected(net, 8)
    # net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    # net = tflearn.regression(net)
    #
    # model = tflearn.DNN(net)
    #
    # model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    # model.save('model.tflearn')

    # Convert your training and output to NumPy arrays (if not already)
    training = numpy.array(training)
    output = numpy.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)

# Define the model
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

try:
    model = load_model("model.keras")
    print("Loaded existing model.")
except Exception as e:
    print("Failed to load model:", e)
    # Train and save as before
    model.fit(training, output, epochs=1000, batch_size=8, verbose=1)
    model.save("model.keras")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("Start talking to the chatbot (Type quit to exit)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(numpy.array([bag_of_words(inp, words)]))
        # print(results)
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        # print(tag)

        if results[0][results_index] > 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))
        else:
            print("I didn't understand that, try again.")

chat()




