import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tqdm import tqdm

# Load the CSV file
csv_path = 'C:/Users/Chanvitha/Downloads/D/sinhala_news_articles.csv'
df = pd.read_csv(csv_path)

# Preprocess the text data
texts = df['text'].values
# labels = df['label'].values
# print(texts[0])

# Perform label encoding for multiclass classification
label_to_int = {
    "Dailymirror_SL": 0,
    "colombotelegrap": 1,
    "NewsfirstSL": 2,
    "theisland_lk__": 3,
    "CeylonToday": 4,
    "NewsWireLK ": 5,
    "colombogazette__ ": 6,
    "TheMorningLK_": 7
}

# Map the labels to their integer representations
labels = df['label'].map(label_to_int)

# Split the data into training and testing sets
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# **********************************************************************************************************************
# Handle non-finite values in the label data
labels_train[np.isnan(labels_train)] = -2  # Replace NaN with 0
labels_train[np.isinf(labels_train)] = -1  # Replace inf with -1
labels_test[np.isnan(labels_test)] = -2  # Replace NaN with 0
labels_test[np.isinf(labels_test)] = -1  # Replace inf with -1

# Preprocess the label data to ensure it's in integer format
labels_train = labels_train.astype(int)  # Convert to integer
labels_test = labels_test.astype(int)  # Convert to integer

# **********************************************************************************************************************

# Tokenize the text data and create word index
num_words = 20000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(texts_train)

sequences_train = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)

# Pad sequences to have the same length
max_sequence_length = 500
X_train = pad_sequences(sequences_train, maxlen=max_sequence_length)
X_test = pad_sequences(sequences_test, maxlen=max_sequence_length)

# Convert integer labels to categorical format
num_classes = 10
y_train = to_categorical(labels_train, num_classes=num_classes)
y_test = to_categorical(labels_test, num_classes=num_classes)

# Load GloVe embeddings
glove_path = 'C:/Users/Chanvitha/Downloads/N/glove.6B.100d.txt'
embedding_dim = 100
embedding_index = {}
with open(glove_path, encoding='utf-8') as f:
    for line in tqdm(f, desc="Loading GloVe", unit=" lines"):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Create embedding matrix
word_index = tokenizer.word_index
num_words = min(num_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Build the 1D CNN model for multiclass classification
model = Sequential()
model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 10
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# predict a text
sample_text = ["""An extraordinary gazette notification was issued considering the necessity of the services provided by any Public Corporation or Government as essential public services.
Accordingly, power supply, distribution of petroleum products and fuel will be deemed as essential public services. All service, work, labour, of any description whatever, necessary to be done in connection with the maintenance, reception, care feeding, treatment of patients in hospitals, nursing homes, dispensaries and other similar institutions.
The gazette notification was issued by President's Secretary E. M. S. B. Ekanayake. (Chaturanga Samarawickrama)"""]
sequences = tokenizer.texts_to_sequences(sample_text)
data = pad_sequences(sequences, maxlen=max_sequence_length)
predictions = model.predict(data)
# Get the predicted class (index of the highest probability)
predicted_class_index = np.argmax(predictions[0])
# Map the index back to the class label
int_to_label = {v: k for k, v in label_to_int.items()}
predicted_class = int_to_label[predicted_class_index]


# print f1 score
from sklearn.metrics import f1_score
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
print("f1_score: ", f1_score(labels_test, y_pred, average='macro'))

# print precision and recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
precision = precision_score(labels_test, y_pred, average='macro')
recall = recall_score(labels_test, y_pred, average='macro')
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print("Sample Text:", sample_text)
print("Predicted Class:", predicted_class)


