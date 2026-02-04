#!/usr/bin/env python3
"""
MazeSolver Neural Network Training Script
Trains a small NN to predict safe motor outputs from sensor readings
Exports weights as C header file for Teensy 4.1

Usage:
    1. Copy CSV files from SD card to this folder
    2. Run: python train_nn.py
    3. Upload generated bot_nn.ino to Teensy
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# Check for TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("Installing TensorFlow...")
    os.system("pip install tensorflow")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

# ==================== CONFIGURATION ====================
DATA_DIR = Path(__file__).parent
MODEL_NAME = "maze_nn"
INPUT_FEATURES = 5  # front, left, right, enc_l_delta, enc_r_delta
OUTPUT_SIZE = 2     # pwm_left, pwm_right

# Neural Network Architecture
HIDDEN_LAYERS = [16, 12, 8]  # Small enough for Teensy
ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'linear'

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Normalization ranges (from your sensor data)
SENSOR_MAX = 4095.0  # 12-bit ADC
ENCODER_MAX = 50.0   # Max encoder ticks per sample
PWM_MAX = 255.0

# ==================== DATA LOADING ====================
def load_all_csvs():
    """Load all run_*.csv files from the data directory"""
    csv_files = sorted(glob.glob(str(DATA_DIR / "run_*.csv")))
    
    if not csv_files:
        print("No run_*.csv files found!")
        print(f"Looking in: {DATA_DIR}")
        print("\nPlease copy CSV files from SD card to this folder.")
        return None
    
    print(f"Found {len(csv_files)} CSV files:")
    
    all_data = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            print(f"  {os.path.basename(f)}: {len(df)} samples")
            all_data.append(df)
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal samples: {len(combined)}")
    return combined

# ==================== DATA PREPROCESSING ====================
def preprocess_data(df):
    """Prepare data for training"""
    
    # Remove rows where robot was stopped (action=3) or crashed (action=4)
    # We want to learn from GOOD driving behavior
    df_clean = df[df['action'].isin([0, 1, 2])].copy()
    
    # Also remove rows where both PWM are 0 (stopped)
    df_clean = df_clean[(df_clean['pwm_l'] > 0) | (df_clean['pwm_r'] > 0)]
    
    print(f"Samples after filtering: {len(df_clean)} ({len(df_clean)/len(df)*100:.1f}%)")
    
    # Input features: normalized sensor readings
    X = np.column_stack([
        df_clean['front'].values / SENSOR_MAX,
        df_clean['left'].values / SENSOR_MAX,
        df_clean['right'].values / SENSOR_MAX,
        np.clip(df_clean['enc_l'].values / ENCODER_MAX, -1, 1),
        np.clip(df_clean['enc_r'].values / ENCODER_MAX, -1, 1),
    ])
    
    # Output: normalized PWM values
    Y = np.column_stack([
        df_clean['pwm_l'].values / PWM_MAX,
        df_clean['pwm_r'].values / PWM_MAX,
    ])
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Y.shape}")
    
    # Data statistics
    print("\nInput statistics:")
    print(f"  Front:  min={X[:,0].min():.3f}, max={X[:,0].max():.3f}, mean={X[:,0].mean():.3f}")
    print(f"  Left:   min={X[:,1].min():.3f}, max={X[:,1].max():.3f}, mean={X[:,1].mean():.3f}")
    print(f"  Right:  min={X[:,2].min():.3f}, max={X[:,2].max():.3f}, mean={X[:,2].mean():.3f}")
    
    return X, Y

# ==================== MODEL BUILDING ====================
def build_model():
    """Build a small dense neural network"""
    model = keras.Sequential([
        layers.Input(shape=(INPUT_FEATURES,)),
    ])
    
    for units in HIDDEN_LAYERS:
        model.add(layers.Dense(units, activation=ACTIVATION))
    
    model.add(layers.Dense(OUTPUT_SIZE, activation=OUTPUT_ACTIVATION))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    model.summary()
    return model

# ==================== TRAINING ====================
def train_model(model, X, Y):
    """Train the model"""
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    Y = Y[indices]
    
    # Train
    history = model.fit(
        X, Y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
    )
    
    # Final metrics
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"\nFinal Loss: {final_loss:.6f}")
    print(f"Final Val Loss: {final_val_loss:.6f}")
    
    return history

# ==================== EXPORT TO C HEADER ====================
def export_to_header(model, filename="nn_weights.h"):
    """Export model weights as C header file for Teensy"""
    
    weights = model.get_weights()
    
    with open(DATA_DIR / filename, 'w') as f:
        f.write("/*\n")
        f.write(" * Neural Network Weights - Auto-generated\n")
        f.write(f" * Architecture: {INPUT_FEATURES} -> {' -> '.join(map(str, HIDDEN_LAYERS))} -> {OUTPUT_SIZE}\n")
        f.write(f" * Activation: {ACTIVATION}\n")
        f.write(" */\n\n")
        f.write("#ifndef NN_WEIGHTS_H\n")
        f.write("#define NN_WEIGHTS_H\n\n")
        
        # Network architecture constants
        f.write(f"#define NN_INPUT_SIZE {INPUT_FEATURES}\n")
        f.write(f"#define NN_OUTPUT_SIZE {OUTPUT_SIZE}\n")
        f.write(f"#define NN_NUM_LAYERS {len(HIDDEN_LAYERS) + 1}\n")
        f.write(f"#define NN_ACTIVATION_RELU 1\n\n")
        
        # Normalization constants
        f.write(f"#define SENSOR_MAX {SENSOR_MAX}f\n")
        f.write(f"#define ENCODER_MAX {ENCODER_MAX}f\n")
        f.write(f"#define PWM_MAX {PWM_MAX}f\n\n")
        
        # Layer sizes array
        layer_sizes = [INPUT_FEATURES] + HIDDEN_LAYERS + [OUTPUT_SIZE]
        f.write(f"const int NN_LAYER_SIZES[{len(layer_sizes)}] = {{")
        f.write(", ".join(map(str, layer_sizes)))
        f.write("};\n\n")
        
        # Export weights and biases for each layer
        layer_idx = 0
        for i in range(0, len(weights), 2):
            W = weights[i]      # weights matrix
            b = weights[i + 1]  # bias vector
            
            rows, cols = W.shape
            
            # Weights
            f.write(f"// Layer {layer_idx}: {rows} inputs -> {cols} outputs\n")
            f.write(f"const float W{layer_idx}[{rows}][{cols}] = {{\n")
            for row in range(rows):
                f.write("  {")
                f.write(", ".join(f"{W[row, col]:.6f}f" for col in range(cols)))
                f.write("},\n")
            f.write("};\n\n")
            
            # Biases
            f.write(f"const float B{layer_idx}[{cols}] = {{")
            f.write(", ".join(f"{b[j]:.6f}f" for j in range(cols)))
            f.write("};\n\n")
            
            layer_idx += 1
        
        f.write("#endif // NN_WEIGHTS_H\n")
    
    print(f"\nWeights exported to: {filename}")
    
    # Calculate model size
    total_params = sum(np.prod(w.shape) for w in weights)
    size_bytes = total_params * 4  # float32
    print(f"Total parameters: {total_params}")
    print(f"Approximate size: {size_bytes} bytes ({size_bytes/1024:.1f} KB)")

# ==================== GENERATE INFERENCE CODE ====================
def generate_inference_code():
    """Generate the bot code with NN inference"""
    
    code = '''/*
 * MazeSolver v5.0 - Neural Network Controlled
 * Uses trained NN for wall avoidance
 * 
 * The NN predicts optimal PWM values based on sensor readings
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include "nn_weights.h"

// ==================== PIN DEFINITIONS ====================
#define PWMA 9
#define PWMB 10
#define AIN1 4
#define AIN2 3
#define BIN1 6
#define BIN2 7
#define STBY 5
#define BTN_PIN 11

const uint8_t SH_F = 21, SH_L = 23, SH_R = 22;

// Encoders
#define ENC_L_A 2
#define ENC_L_B 3
#define ENC_R_A 4
#define ENC_R_B 5

Adafruit_SH1106G display(128, 64, &Wire, -1);

// ==================== ENCODER STATE ====================
volatile int32_t encCountL = 0, encCountR = 0;
int32_t lastEncL = 0, lastEncR = 0;

// ==================== SENSOR BUFFERS ====================
#define BUF_SIZE 5
int16_t fBuf[BUF_SIZE], lBuf[BUF_SIZE], rBuf[BUF_SIZE];
uint8_t bufIdx = 0;
int16_t sF = 0, sL = 0, sR = 0;

// ==================== NN BUFFERS ====================
float nnInput[NN_INPUT_SIZE];
float nnHidden[3][16];  // Max hidden layer size
float nnOutput[NN_OUTPUT_SIZE];

// ==================== STATE ====================
volatile bool running = false;
const int16_t BASE_SPD = 130;
const int16_t MIN_SPD = 55;
const int16_t MAX_SPD = 170;

// Thresholds for hard-coded safety backup
const int16_t FRONT_EMERGENCY = 2700;
const int16_t SIDE_CRASH = 3100;

// ==================== ENCODER ISRs ====================
void encLeftISR() {
  if (digitalRead(ENC_L_B)) encCountL++;
  else encCountL--;
}

void encRightISR() {
  if (digitalRead(ENC_R_B)) encCountR++;
  else encCountR--;
}

// ==================== MEDIAN FILTER ====================
int16_t median5(int16_t* b) {
  int16_t t[BUF_SIZE];
  memcpy(t, b, sizeof(t));
  for (uint8_t i = 1; i < BUF_SIZE; i++) {
    int16_t key = t[i];
    int8_t j = i - 1;
    while (j >= 0 && t[j] > key) {
      t[j + 1] = t[j];
      j--;
    }
    t[j + 1] = key;
  }
  return t[2];
}

void readSensors() {
  fBuf[bufIdx] = analogRead(SH_F);
  lBuf[bufIdx] = analogRead(SH_L);
  rBuf[bufIdx] = analogRead(SH_R);
  bufIdx = (bufIdx + 1) % BUF_SIZE;
  sF = median5(fBuf);
  sL = median5(lBuf);
  sR = median5(rBuf);
}

// ==================== NEURAL NETWORK INFERENCE ====================
inline float relu(float x) {
  return x > 0 ? x : 0;
}

void nnInference() {
  // Prepare normalized input
  int16_t deltaEncL = encCountL - lastEncL;
  int16_t deltaEncR = encCountR - lastEncR;
  lastEncL = encCountL;
  lastEncR = encCountR;
  
  nnInput[0] = (float)sF / SENSOR_MAX;
  nnInput[1] = (float)sL / SENSOR_MAX;
  nnInput[2] = (float)sR / SENSOR_MAX;
  nnInput[3] = constrain((float)deltaEncL / ENCODER_MAX, -1.0f, 1.0f);
  nnInput[4] = constrain((float)deltaEncR / ENCODER_MAX, -1.0f, 1.0f);
  
  // Layer 0: Input -> Hidden0 (16 neurons)
  for (int j = 0; j < 16; j++) {
    float sum = B0[j];
    for (int i = 0; i < 5; i++) {
      sum += nnInput[i] * W0[i][j];
    }
    nnHidden[0][j] = relu(sum);
  }
  
  // Layer 1: Hidden0 -> Hidden1 (12 neurons)
  for (int j = 0; j < 12; j++) {
    float sum = B1[j];
    for (int i = 0; i < 16; i++) {
      sum += nnHidden[0][i] * W1[i][j];
    }
    nnHidden[1][j] = relu(sum);
  }
  
  // Layer 2: Hidden1 -> Hidden2 (8 neurons)
  for (int j = 0; j < 8; j++) {
    float sum = B2[j];
    for (int i = 0; i < 12; i++) {
      sum += nnHidden[1][i] * W2[i][j];
    }
    nnHidden[2][j] = relu(sum);
  }
  
  // Layer 3: Hidden2 -> Output (2 neurons, linear)
  for (int j = 0; j < 2; j++) {
    float sum = B3[j];
    for (int i = 0; i < 8; i++) {
      sum += nnHidden[2][i] * W3[i][j];
    }
    nnOutput[j] = sum;
  }
}

// ==================== MOTOR CONTROL ====================
void setMotors(int16_t L, int16_t R) {
  L = constrain(L, 0, 255);
  R = constrain(R, 0, 255);
  if (L > 0 && L < MIN_SPD) L = MIN_SPD;
  if (R > 0 && R < MIN_SPD) R = MIN_SPD;
  analogWrite(PWMA, L);
  analogWrite(PWMB, R);
}

void stopMotors() {
  analogWrite(PWMA, 0);
  analogWrite(PWMB, 0);
}

void setForward() {
  digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW);
}

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  
  pinMode(BTN_PIN, INPUT_PULLUP);
  pinMode(STBY, OUTPUT);
  pinMode(AIN1, OUTPUT); pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT); pinMode(BIN2, OUTPUT);
  pinMode(PWMA, OUTPUT); pinMode(PWMB, OUTPUT);
  
  pinMode(ENC_L_A, INPUT_PULLUP);
  pinMode(ENC_L_B, INPUT_PULLUP);
  pinMode(ENC_R_A, INPUT_PULLUP);
  pinMode(ENC_R_B, INPUT_PULLUP);
  
  attachInterrupt(digitalPinToInterrupt(ENC_L_A), encLeftISR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_R_A), encRightISR, RISING);
  
  digitalWrite(STBY, HIGH);
  analogReadResolution(12);
  setForward();
  
  memset(fBuf, 0, sizeof(fBuf));
  memset(lBuf, 0, sizeof(lBuf));
  memset(rBuf, 0, sizeof(rBuf));
  
  display.begin(0x3C, true);
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 8);
  display.println("MazeSolver v5.0");
  display.setCursor(0, 24);
  display.println("Neural Network Mode");
  display.setCursor(0, 48);
  display.println("BTN to START");
  display.display();
  
  Serial.println("=== MazeSolver v5.0 - NN Mode ===");
}

// ==================== MAIN LOOP ====================
void loop() {
  static uint32_t lastBtn = 0;
  static uint32_t lastDisplay = 0;
  
  // Button handling
  if (digitalRead(BTN_PIN) == LOW && millis() - lastBtn > 300) {
    lastBtn = millis();
    running = !running;
    if (!running) stopMotors();
    delay(150);
  }
  
  if (!running) {
    display.clearDisplay();
    display.setCursor(0, 20);
    display.println("=== PAUSED ===");
    display.setCursor(0, 40);
    display.println("BTN to START");
    display.display();
    stopMotors();
    delay(80);
    return;
  }
  
  // Read sensors
  readSensors();
  
  // Run neural network
  nnInference();
  
  // Convert NN output to PWM (denormalize)
  int16_t pwmL = (int16_t)(nnOutput[0] * PWM_MAX);
  int16_t pwmR = (int16_t)(nnOutput[1] * PWM_MAX);
  
  // Safety override: hard-coded emergency stop
  if (sF > FRONT_EMERGENCY) {
    stopMotors();
    Serial.println("! EMERGENCY - Front too close");
    delay(100);
    return;
  }
  
  // Safety override: side crash avoidance
  if (sL > SIDE_CRASH) {
    pwmL += 50;
    pwmR -= 20;
  }
  if (sR > SIDE_CRASH) {
    pwmL -= 20;
    pwmR += 50;
  }
  
  // Constrain and apply
  pwmL = constrain(pwmL, MIN_SPD, MAX_SPD);
  pwmR = constrain(pwmR, MIN_SPD, MAX_SPD);
  setMotors(pwmL, pwmR);
  
  // Debug output
  Serial.print("F:");
  Serial.print(sF);
  Serial.print(" L:");
  Serial.print(sL);
  Serial.print(" R:");
  Serial.print(sR);
  Serial.print(" -> PWM L:");
  Serial.print(pwmL);
  Serial.print(" R:");
  Serial.println(pwmR);
  
  // Display update
  if (millis() - lastDisplay > 150) {
    lastDisplay = millis();
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("NN MODE RUNNING");
    display.setCursor(0, 14);
    display.print("F:");
    display.print(sF);
    display.setCursor(0, 26);
    display.print("L:");
    display.print(sL);
    display.print(" R:");
    display.print(sR);
    display.setCursor(0, 40);
    display.print("PWM L:");
    display.print(pwmL);
    display.print(" R:");
    display.print(pwmR);
    display.display();
  }
  
  delay(20);  // 50Hz
}
'''
    
    with open(DATA_DIR / "bot_nn.ino", 'w') as f:
        f.write(code)
    
    print(f"Generated: bot_nn.ino")

# ==================== MAIN ====================
def main():
    print("=" * 50)
    print("MazeSolver Neural Network Training")
    print("=" * 50)
    
    # Load data
    df = load_all_csvs()
    
    if df is None:
        # Create sample data for testing
        print("\nCreating sample training data for demo...")
        np.random.seed(42)
        n = 1000
        
        # Simulate sensor readings and good PWM responses
        front = np.random.randint(500, 3000, n)
        left = np.random.randint(800, 3200, n)
        right = np.random.randint(800, 3200, n)
        enc_l = np.random.randint(-10, 30, n)
        enc_r = np.random.randint(-10, 30, n)
        
        # Simulate good driving: when front is high (close), slow down
        # When left/right imbalanced, correct
        base_pwm = 120
        pwm_l = np.clip(base_pwm - (front - 1500) * 0.02 - (left - right) * 0.01, 50, 180)
        pwm_r = np.clip(base_pwm - (front - 1500) * 0.02 + (left - right) * 0.01, 50, 180)
        
        df = pd.DataFrame({
            'time_ms': range(n),
            'front': front,
            'left': left,
            'right': right,
            'enc_l': enc_l,
            'enc_r': enc_r,
            'pwm_l': pwm_l.astype(int),
            'pwm_r': pwm_r.astype(int),
            'action': np.zeros(n, dtype=int)
        })
        
        print(f"Created {n} sample data points")
    
    # Preprocess
    X, Y = preprocess_data(df)
    
    # Build model
    print("\nBuilding model...")
    model = build_model()
    
    # Train
    print("\nTraining...")
    train_model(model, X, Y)
    
    # Export weights
    print("\nExporting weights...")
    export_to_header(model)
    
    # Generate inference code
    print("\nGenerating inference code...")
    generate_inference_code()
    
    # Save model
    model.save(DATA_DIR / "maze_nn_model.keras")
    print(f"Model saved: maze_nn_model.keras")
    
    print("\n" + "=" * 50)
    print("DONE!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Copy CSV files from SD card to this folder")
    print("2. Re-run this script with real data")
    print("3. Upload 'bot_nn.ino' + 'nn_weights.h' to Teensy")
    print("=" * 50)

if __name__ == "__main__":
    main()
