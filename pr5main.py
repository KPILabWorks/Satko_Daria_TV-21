import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("bbc-text.csv")

label_encoder = LabelEncoder()
df["category"] = label_encoder.fit_transform(df["category"])

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].values, df["category"].values, test_size=0.2, random_state=42
)

max_tokens = 10000
max_len = 200

vectorizer = TextVectorization(max_tokens=max_tokens, output_mode='int', output_sequence_length=max_len)
vectorizer.adapt(train_texts)

model = keras.Sequential([
    vectorizer,
    Embedding(input_dim=max_tokens, output_dim=128),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dropout(0.5),
    Dense(32, activation='sigmoid'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_texts, train_labels, validation_data=(test_texts, test_labels), epochs=20, batch_size=32)

test_loss, test_acc = model.evaluate(test_texts, test_labels)
print(f"Точність на тестових даних: {test_acc:.4f}")
