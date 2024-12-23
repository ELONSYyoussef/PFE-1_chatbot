import os
import nltk
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
from tkinter import filedialog
from nltk.stem import WordNetLemmatizer

# Télécharger les packages nécessaires de NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Fonction pour sélectionner le répertoire contenant les fichiers nécessaires
def select_directory():
    directory = filedialog.askdirectory()
    return directory

# Sélectionner le répertoire contenant les fichiers nécessaires
directory = select_directory()
if not directory:
    print("Vous devez sélectionner un répertoire.")
    exit()

intents_path = os.path.join(directory, 'intents.json')
words_path = os.path.join(directory, 'words.pkl')
classes_path = os.path.join(directory, 'classes.pkl')
model_path = os.path.join(directory, 'chatbot_model.h5')

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open(intents_path).read()
intents = json.loads(data_file)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, intent['tag']))

        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sort classes
classes = sorted(list(set(classes)))

# Documents = combination between patterns and intents
print(len(documents), "documents")
# Classes = intents
print(len(classes), "classes", classes)
# Words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

# Sauvegarder les mots et les classes dans des fichiers pickle
pickle.dump(words, open(words_path, 'wb'))
pickle.dump(classes, open(classes_path, 'wb'))

# Préprocesser les données et créer l'ensemble d'entraînement
training = []
output_empty = [0] * len(classes)

for doc in documents:
    # Initialize bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize each word
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create the bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # Append bag of words and output row to training list
    training.append((bag, output_row))

# Shuffle the features and make numpy array
random.shuffle(training)

# Split the features and labels
X_train = np.array([item[0] for item in training])
Y_train = np.array([item[1] for item in training])

print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
# Add input layer
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(Y_train[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(momentum=0.9, nesterov=True)  # Specify momentum and nesterov only
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting and saving the model
hist = model.fit(np.array(X_train), np.array(Y_train), epochs=200, batch_size=5, verbose=1)
model.save(model_path, hist)

print("Model created")
