#!/usr/bin/env python3
"""
MazeSolver Neural Network Training - FINAL
===========================================
Trains on collected data + injects safety samples
Guarantees correct steering direction for wall avoidance

Your Robot: 120mm x 140mm, 250mm track
Sensor Calibration:
  - Front: 3000 @ 3cm, 2000 @ 2cm
  - Sides: 4000 touching, ~2000 centered
"""

import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path

# ==================== CONFIGURATION ====================
DATA_DIR = Path(__file__).parent
CSV_PATTERN = "maze_*.csv"

# Neural Network Architecture
INPUT_SIZE = 3    # front, left, right
HIDDEN1 = 32
HIDDEN2 = 16
OUTPUT_SIZE = 2   # pwm_left, pwm_right

# Training parameters
LEARNING_RATE = 0.001
EPOCHS = 500
BATCH_SIZE = 64

# Normalization constants (matching your sensor range)
SCALE_SENSOR = 4095.0  # 12-bit ADC
SCALE_PWM = 255.0

# ==================== LOAD DATA ====================
def load_data():
    """Load all CSV files and combine"""
    csv_files = sorted(glob.glob(str(DATA_DIR / CSV_PATTERN)))
    print(f"Found {len(csv_files)} CSV files")
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  {os.path.basename(f)}: {len(df)} samples")
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    
    data = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal samples: {len(data)}")
    return data

# ==================== SAFETY INJECTION ====================
def inject_safety_samples(n_per_rule=2000):
    """
    Inject synthetic samples that GUARANTEE correct steering.
    These override any bad patterns in the training data.
    
    CRITICAL STEERING LOGIC:
    - Close to LEFT wall  -> LEFT wheel faster  -> turns RIGHT (away from left)
    - Close to RIGHT wall -> RIGHT wheel faster -> turns LEFT (away from right)
    """
    samples = []
    
    print("\nInjecting safety samples...")
    
    # Rule 1: DANGER LEFT - very close to left wall
    # Left sensor high (3200-3800), right low -> HARD RIGHT (left fast, right slow)
    for _ in range(n_per_rule):
        f = np.random.uniform(400, 1500)
        l = np.random.uniform(3200, 3800)  # DANGER zone
        r = np.random.uniform(1500, 2500)
        samples.append([f, l, r, 170, 70])  # Hard right turn
    print(f"  DANGER_LEFT: {n_per_rule} samples -> PWM[170,70]")
    
    # Rule 2: DANGER RIGHT - very close to right wall
    for _ in range(n_per_rule):
        f = np.random.uniform(400, 1500)
        l = np.random.uniform(1500, 2500)
        r = np.random.uniform(3200, 3800)  # DANGER zone
        samples.append([f, l, r, 70, 170])  # Hard left turn
    print(f"  DANGER_RIGHT: {n_per_rule} samples -> PWM[70,170]")
    
    # Rule 3: CLOSE LEFT - moderately close to left
    for _ in range(n_per_rule):
        f = np.random.uniform(400, 1800)
        l = np.random.uniform(2600, 3200)  # Close zone
        r = np.random.uniform(1500, 2200)
        samples.append([f, l, r, 145, 115])  # Mild right
    print(f"  CLOSE_LEFT: {n_per_rule} samples -> PWM[145,115]")
    
    # Rule 4: CLOSE RIGHT - moderately close to right
    for _ in range(n_per_rule):
        f = np.random.uniform(400, 1800)
        l = np.random.uniform(1500, 2200)
        r = np.random.uniform(2600, 3200)  # Close zone
        samples.append([f, l, r, 115, 145])  # Mild left
    print(f"  CLOSE_RIGHT: {n_per_rule} samples -> PWM[115,145]")
    
    # Rule 5: CENTERED - balanced sensors
    for _ in range(n_per_rule):
        f = np.random.uniform(400, 1800)
        base = np.random.uniform(1800, 2400)
        diff = np.random.uniform(-200, 200)
        l = base + diff
        r = base - diff
        samples.append([f, l, r, 130, 130])  # Straight
    print(f"  CENTERED: {n_per_rule} samples -> PWM[130,130]")
    
    # Rule 6: FRONT BLOCKED - wall ahead
    for _ in range(n_per_rule):
        f = np.random.uniform(2600, 3500)  # Wall ahead
        l = np.random.uniform(1500, 2800)
        r = np.random.uniform(1500, 2800)
        samples.append([f, l, r, 0, 0])  # Stop
    print(f"  FRONT_BLOCKED: {n_per_rule} samples -> PWM[0,0]")
    
    # Rule 7: TURN LEFT - opening on left
    for _ in range(n_per_rule):
        f = np.random.uniform(400, 1800)
        l = np.random.uniform(300, 1000)   # Open left
        r = np.random.uniform(2000, 3000)  # Wall right
        samples.append([f, l, r, 70, 130])  # Turn left
    print(f"  TURN_LEFT: {n_per_rule} samples -> PWM[70,130]")
    
    # Rule 8: TURN RIGHT - opening on right
    for _ in range(n_per_rule):
        f = np.random.uniform(400, 1800)
        l = np.random.uniform(2000, 3000)  # Wall left
        r = np.random.uniform(300, 1000)   # Open right
        samples.append([f, l, r, 130, 70])  # Turn right
    print(f"  TURN_RIGHT: {n_per_rule} samples -> PWM[130,70]")
    
    # Rule 9: OPEN SPACE - no walls
    for _ in range(n_per_rule // 2):
        f = np.random.uniform(300, 1000)
        l = np.random.uniform(300, 1000)
        r = np.random.uniform(300, 1000)
        samples.append([f, l, r, 140, 140])  # Cruise
    print(f"  OPEN_SPACE: {n_per_rule//2} samples -> PWM[140,140]")
    
    # Rule 10: CRASH PREVENTION - extreme danger
    for _ in range(n_per_rule):
        f = np.random.uniform(400, 1200)
        l = np.random.uniform(3600, 4000)  # Almost touching!
        r = np.random.uniform(1200, 2000)
        samples.append([f, l, r, 180, 55])  # Emergency right
    print(f"  CRASH_LEFT: {n_per_rule} samples -> PWM[180,55]")
    
    for _ in range(n_per_rule):
        f = np.random.uniform(400, 1200)
        l = np.random.uniform(1200, 2000)
        r = np.random.uniform(3600, 4000)  # Almost touching!
        samples.append([f, l, r, 55, 180])  # Emergency left
    print(f"  CRASH_RIGHT: {n_per_rule} samples -> PWM[55,180]")
    
    arr = np.array(samples)
    print(f"\nTotal safety samples: {len(arr)}")
    return arr

# ==================== NEURAL NETWORK ====================
class NeuralNetwork:
    def __init__(self):
        # Xavier initialization
        self.w1 = np.random.randn(HIDDEN1, INPUT_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
        self.b1 = np.zeros(HIDDEN1)
        self.w2 = np.random.randn(HIDDEN2, HIDDEN1) * np.sqrt(2.0 / HIDDEN1)
        self.b2 = np.zeros(HIDDEN2)
        self.w3 = np.random.randn(OUTPUT_SIZE, HIDDEN2) * np.sqrt(2.0 / HIDDEN2)
        self.b3 = np.zeros(OUTPUT_SIZE)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.z1 = X @ self.w1.T + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.w2.T + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.w3.T + self.b3
        return self.z3  # Linear output
    
    def backward(self, X, y, output, lr):
        m = X.shape[0]
        
        # Output layer
        dz3 = (output - y) / m
        dw3 = dz3.T @ self.a2
        db3 = np.sum(dz3, axis=0)
        
        # Hidden layer 2
        da2 = dz3 @ self.w3
        dz2 = da2 * self.relu_derivative(self.z2)
        dw2 = dz2.T @ self.a1
        db2 = np.sum(dz2, axis=0)
        
        # Hidden layer 1
        da1 = dz2 @ self.w2
        dz1 = da1 * self.relu_derivative(self.z1)
        dw1 = dz1.T @ X
        db1 = np.sum(dz1, axis=0)
        
        # Update weights
        self.w3 -= lr * dw3
        self.b3 -= lr * db3
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
    
    def train(self, X, y, epochs, batch_size, lr):
        n_samples = X.shape[0]
        history = []
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                output = self.forward(X_batch)
                loss = np.mean((output - y_batch) ** 2)
                epoch_loss += loss
                n_batches += 1
                
                self.backward(X_batch, y_batch, output, lr)
            
            avg_loss = epoch_loss / n_batches
            history.append(avg_loss)
            
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d}: Loss = {avg_loss:.6f}")
        
        return history

# ==================== EXPORT WEIGHTS ====================
def export_weights(nn, filename="weights.h"):
    """Export weights to C header file for Teensy"""
    
    with open(DATA_DIR / filename, 'w') as f:
        f.write("// Auto-generated weights for MazeSolver NN\n")
        f.write(f"// Architecture: {INPUT_SIZE} -> {HIDDEN1} -> {HIDDEN2} -> {OUTPUT_SIZE}\n")
        f.write(f"// Activation: ReLU (hidden), Linear (output)\n\n")
        
        f.write(f"#define SCALE_SENSOR {SCALE_SENSOR:.1f}f\n")
        f.write(f"#define SCALE_PWM {SCALE_PWM:.1f}f\n\n")
        
        # Layer 1: w1[32][3], b1[32]
        f.write(f"const float w1[{HIDDEN1}][{INPUT_SIZE}] = {{\n")
        for i in range(HIDDEN1):
            row = ", ".join(f"{v:.6f}f" for v in nn.w1[i])
            f.write(f"  {{{row}}}")
            f.write(",\n" if i < HIDDEN1-1 else "\n")
        f.write("};\n\n")
        
        f.write(f"const float b1[{HIDDEN1}] = {{")
        f.write(", ".join(f"{v:.6f}f" for v in nn.b1))
        f.write("};\n\n")
        
        # Layer 2: w2[16][32], b2[16]
        f.write(f"const float w2[{HIDDEN2}][{HIDDEN1}] = {{\n")
        for i in range(HIDDEN2):
            row = ", ".join(f"{v:.6f}f" for v in nn.w2[i])
            f.write(f"  {{{row}}}")
            f.write(",\n" if i < HIDDEN2-1 else "\n")
        f.write("};\n\n")
        
        f.write(f"const float b2[{HIDDEN2}] = {{")
        f.write(", ".join(f"{v:.6f}f" for v in nn.b2))
        f.write("};\n\n")
        
        # Layer 3: w3[2][16], b3[2]
        f.write(f"const float w3[{OUTPUT_SIZE}][{HIDDEN2}] = {{\n")
        for i in range(OUTPUT_SIZE):
            row = ", ".join(f"{v:.6f}f" for v in nn.w3[i])
            f.write(f"  {{{row}}}")
            f.write(",\n" if i < OUTPUT_SIZE-1 else "\n")
        f.write("};\n\n")
        
        f.write(f"const float b3[{OUTPUT_SIZE}] = {{")
        f.write(", ".join(f"{v:.6f}f" for v in nn.b3))
        f.write("};\n")
    
    print(f"\nWeights saved to {filename}")
    total_params = HIDDEN1*(INPUT_SIZE+1) + HIDDEN2*(HIDDEN1+1) + OUTPUT_SIZE*(HIDDEN2+1)
    print(f"Total parameters: {total_params}")

# ==================== VALIDATION ====================
def validate_steering(nn):
    """Test that steering direction is correct"""
    print("\n" + "="*50)
    print("STEERING VALIDATION")
    print("="*50)
    
    tests = [
        # (front, left, right, description, expected_behavior)
        (500, 3500, 2000, "DANGER LEFT",  "L > R (turn right)"),
        (500, 2000, 3500, "DANGER RIGHT", "R > L (turn left)"),
        (500, 2000, 2000, "CENTERED",     "L ≈ R (straight)"),
        (500, 2800, 2000, "CLOSE LEFT",   "L > R (mild right)"),
        (500, 2000, 2800, "CLOSE RIGHT",  "R > L (mild left)"),
        (3000, 2000, 2000, "FRONT BLOCK", "Both low (stop)"),
        (500, 600, 2500, "TURN LEFT",     "R > L (turn left)"),
        (500, 2500, 600, "TURN RIGHT",    "L > R (turn right)"),
    ]
    
    all_pass = True
    for f, l, r, desc, expected in tests:
        X = np.array([[f/SCALE_SENSOR, l/SCALE_SENSOR, r/SCALE_SENSOR]])
        out = nn.forward(X)[0] * SCALE_PWM
        pwm_l, pwm_r = out[0], out[1]
        
        # Check correctness
        if "DANGER LEFT" in desc or "CLOSE LEFT" in desc or "TURN RIGHT" in desc:
            passed = pwm_l > pwm_r
        elif "DANGER RIGHT" in desc or "CLOSE RIGHT" in desc or "TURN LEFT" in desc:
            passed = pwm_r > pwm_l
        elif "CENTERED" in desc:
            passed = abs(pwm_l - pwm_r) < 30
        elif "FRONT" in desc:
            passed = pwm_l < 80 and pwm_r < 80
        else:
            passed = True
        
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        
        print(f"{desc:15s}: F={f:4d} L={l:4d} R={r:4d} -> PWM[{pwm_l:6.1f}, {pwm_r:6.1f}] {status}")
    
    print("="*50)
    if all_pass:
        print("✅ ALL STEERING TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED - Check training data")
    print("="*50)
    
    return all_pass

# ==================== MAIN ====================
def main():
    print("="*60)
    print("MazeSolver Neural Network Training - FINAL")
    print("="*60)
    
    # Load collected data
    data = load_data()
    
    # Extract features and targets
    # Using ideal_l and ideal_r as targets (what the NN should learn)
    X_real = data[['front', 'left', 'right']].values
    y_real = data[['ideal_l', 'ideal_r']].values
    
    print(f"\nReal data: {len(X_real)} samples")
    
    # Inject safety samples
    safety = inject_safety_samples(n_per_rule=2500)
    X_safety = safety[:, :3]
    y_safety = safety[:, 3:]
    
    # Combine (safety samples are critical, so we add them twice)
    X = np.vstack([X_real, X_safety, X_safety])
    y = np.vstack([y_real, y_safety, y_safety])
    
    print(f"\nCombined dataset: {len(X)} samples")
    
    # Normalize
    X_norm = X / SCALE_SENSOR
    y_norm = y / SCALE_PWM
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X_norm = X_norm[indices]
    y_norm = y_norm[indices]
    
    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    nn = NeuralNetwork()
    history = nn.train(X_norm, y_norm, EPOCHS, BATCH_SIZE, LEARNING_RATE)
    
    print(f"\nFinal Loss: {history[-1]:.6f}")
    
    # Validate
    validate_steering(nn)
    
    # Export
    export_weights(nn)
    
    print("\n✅ Training complete! Upload bot.ino with new weights.h")

if __name__ == "__main__":
    main()
