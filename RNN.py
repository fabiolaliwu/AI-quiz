import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# --- 1. CONFIGURATION AND DATA LOADING ---
# You can replace this with your chosen corpus (Shakespeare, lyrics, etc.)
# For the sprint, start with a small, manageable corpus.
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
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
vocab_size = len(chars)
print(f"Vocabulary size (unique characters): {vocab_size}")

# Define sequence parameters
sequence_length = 50 # The LSTM will look back 50 characters to predict the next one
step = 1 # How many characters to skip when creating sequences (1 = every character)
dataX = []
dataY = []

# Create input and output sequences
for i in range(0, len(raw_text) - sequence_length, step):
    seq_in = raw_text[i:i + sequence_length]
    seq_out = raw_text[i + sequence_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print(f"Total training patterns: {n_patterns}")

# Reshape input X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, sequence_length))

# One-hot encode the output variable y
y = to_categorical(dataY, num_classes=vocab_size)

# --- 3. MODEL DEFINITION (The LSTM) ---

# Hyperparameters
embedding_dim = 128
lstm_units = 512

model = Sequential([
    # Input Layer: Converts character indices into dense vectors
    Embedding(input_dim=vocab_size,
              output_dim=embedding_dim,
              input_length=sequence_length),

    # Recurrent Layer: Processes the sequence and maintains state
    # Set return_sequences=False on the last LSTM layer (default behavior)
    LSTM(lstm_units),

    # Output Layer: Predicts the probability of the next character
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
print(model.summary())

# --- 4. TRAINING (Fit the Model) ---
# Note: For real text generation, you'd need many more epochs and a larger dataset!
# epochs = 50 # Increase this for better results
# model.fit(X, y, epochs=epochs, batch_size=64, verbose=2)

print("\nModel definition complete. Skipping long training for example.")

# --- 5. TEXT GENERATION (Inference) ---

def sample(preds, temperature=1.0):
    """
    Helper function to sample an index from a probability array,
    using a temperature to control creativity.
    """
    # Cast probabilities to float64 for stability
    preds = np.asarray(preds).astype('float64')
    # Apply temperature
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # Sample index
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, seed_text, length=200, temperature=0.7):
    """
    Generates text iteratively using the trained model.
    """
    generated_text = seed_text
    
    # Map the seed text to integers
    pattern = [char_to_int[char] for char in seed_text]
    
    for i in range(length):
        # Reshape the input sequence for the model: (1, sequence_length)
        x = np.reshape(pattern, (1, len(pattern)))
        
        # Get the prediction probabilities
        prediction_probabilities = model.predict(x, verbose=0)[0]
        
        # Sample the next character index
        index = sample(prediction_probabilities, temperature)
        
        # Convert index back to character
        result = int_to_char[index]
        
        # Append to the generated text
        generated_text += result
        
        # Update the input pattern by dropping the first element and adding the new one
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return generated_text

# --- Example of how to use the generator (AFTER training) ---

# Since we skipped training, let's assume you have a trained model.
# To run this, you must run the model.fit() line above.
# print("\n--- Generated Text (Temperature=0.2) ---")
# # Get a random seed sequence to start
# start_index = np.random.randint(0, len(dataX)-1)
# seed = raw_text[start_index:start_index + sequence_length]
# print(generate_text(model, seed, length=100, temperature=0.2))

# print("\n--- Generated Text (Temperature=1.0) ---")
# # Demonstrate higher creativity/randomness
# print(generate_text(model, seed, length=100, temperature=1.0))