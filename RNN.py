import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# --- 1. CONFIGURATION AND DATA LOADING ---
raw_text = """
The quick brown fox jumps over the lazy dog.
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles.
"""

# Convert text to lowercase
raw_text = raw_text.lower()
print(f"Total characters in corpus: {len(raw_text)}")

# --- 2. DATA PREPROCESSING ---

# Create character vocabulary
chars = sorted(list(set(raw_text)))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)
print(f"Vocabulary size (unique characters): {vocab_size}")

# Define sequence parameters
sequence_length = 50  # The LSTM will look back 50 characters
step = 1  # Step size for creating sequences
dataX, dataY = [], []

for i in range(0, len(raw_text) - sequence_length, step):
    seq_in = raw_text[i:i + sequence_length]
    seq_out = raw_text[i + sequence_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print(f"Total training patterns: {n_patterns}")

# Prepare input and output
X = np.array(dataX)
y = to_categorical(dataY, num_classes=vocab_size)

# --- 3. MODEL DEFINITION (LSTM TEXT GENERATOR) ---

embedding_dim = 128
lstm_units = 512

model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_shape=(sequence_length,)
    ),
    LSTM(lstm_units),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.build(input_shape=(None, sequence_length))
print(model.summary())

# --- 4. TRAINING (SHORT DEMO TRAINING LOOP) ---

# Checkpoint to save best weights
checkpoint = ModelCheckpoint('model_weights.h5', save_best_only=True, verbose=1)

print("\nTraining model... (this will be quick for demonstration)")
model.fit(X, y, epochs=5, batch_size=32, verbose=2, callbacks=[checkpoint])
print("Training complete!")

# --- 5. TEXT GENERATION ---

def sample(preds, temperature=1.0):
    """
    Samples an index from a probability array using temperature scaling.
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature  # prevent log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, seed_text, length=200, temperature=0.7):
    """
    Generates new text from a trained LSTM model given a seed string.
    """
    generated_text = seed_text
    pattern = [char_to_int[c] for c in seed_text]

    for i in range(length):
        x = np.reshape(pattern, (1, len(pattern)))
        prediction_probabilities = model.predict(x, verbose=0)[0]
        index = sample(prediction_probabilities, temperature)
        result = int_to_char[index]
        generated_text += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return generated_text

# --- 6. GENERATE SAMPLE TEXTS (AFTER TRAINING) ---

start_index = np.random.randint(0, len(dataX) - 1)
seed = raw_text[start_index:start_index + sequence_length]

print("\n--- Generated Text (Temperature = 0.2, predictable) ---")
print(generate_text(model, seed, length=200, temperature=0.2))

print("\n--- Generated Text (Temperature = 1.0, creative) ---")
print(generate_text(model, seed, length=200, temperature=1.0))
