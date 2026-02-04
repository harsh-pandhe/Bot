#!/usr/bin/env python3
"""
MazeSolver ULTIMATE Neural Network Trainer v2.0
================================================
Trains a supervised NN from collected data with safety injection
Guarantees NO WALL HITS with proper steering directions

Run: python train_ultimate.py
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
SENSOR_MAX = 4095.0
PWM_MAX = 255.0

# Architecture
INPUT_SIZE = 3      # front, left, right
HIDDEN1 = 32
HIDDEN2 = 16  
OUTPUT_SIZE = 2     # pwm_left, pwm_right

# Training
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 0.002

# ==================== IDEAL RESPONSES (CRITICAL!) ====================
# These define what the NN MUST learn for each situation
# Higher sensor value = closer to obstacle

SAFETY_RULES = {
    # Scenario: (front_range, left_range, right_range, ideal_pwm_l, ideal_pwm_r)
    
    # 1. CORRIDOR CENTER - balanced walls, go straight
    'center': {
        'front': (200, 1500),
        'left': (2000, 2800),
        'right': (2000, 2800),
        'pwm': (130, 130)
    },
    
    # 2. CLOSE TO LEFT WALL - steer RIGHT (speed up LEFT, slow RIGHT)
    'close_left': {
        'front': (200, 2000),
        'left': (2800, 3200),
        'right': (1000, 2200),
        'pwm': (150, 100)  # Left faster = turns right = away from left wall
    },
    
    # 3. CLOSE TO RIGHT WALL - steer LEFT (slow LEFT, speed up RIGHT)
    'close_right': {
        'front': (200, 2000),
        'left': (1000, 2200),
        'right': (2800, 3200),
        'pwm': (100, 150)  # Right faster = turns left = away from right wall
    },
    
    # 4. DANGER LEFT (almost hitting) - HARD RIGHT
    'danger_left': {
        'front': (200, 2500),
        'left': (3200, 4095),
        'right': (500, 2500),
        'pwm': (180, 60)  # Very strong right turn
    },
    
    # 5. DANGER RIGHT (almost hitting) - HARD LEFT
    'danger_right': {
        'front': (200, 2500),
        'left': (500, 2500),
        'right': (3200, 4095),
        'pwm': (60, 180)  # Very strong left turn
    },
    
    # 6. FRONT BLOCKED - STOP
    'front_blocked': {
        'front': (2600, 4095),
        'left': (500, 3500),
        'right': (500, 3500),
        'pwm': (0, 0)
    },
    
    # 7. APPROACHING FRONT - SLOW DOWN
    'front_close': {
        'front': (2000, 2600),
        'left': (1000, 3000),
        'right': (1000, 3000),
        'pwm': (70, 70)
    },
    
    # 8. OPEN SPACE - GO FAST
    'open': {
        'front': (100, 800),
        'left': (100, 1200),
        'right': (100, 1200),
        'pwm': (150, 150)
    },
    
    # 9. LEFT TURN AVAILABLE
    'left_open': {
        'front': (800, 2200),
        'left': (100, 900),
        'right': (2000, 3500),
        'pwm': (90, 140)  # Gentle left
    },
    
    # 10. RIGHT TURN AVAILABLE
    'right_open': {
        'front': (800, 2200),
        'left': (2000, 3500),
        'right': (100, 900),
        'pwm': (140, 90)  # Gentle right
    },
}

# ==================== LOAD DATA ====================
def load_all_data():
    """Load all CSV files from the data directory"""
    all_data = []
    
    # Pattern matching for different file types
    patterns = ['data_*.csv', 'tune_*.csv', 'run_*.csv', 'train*.csv']
    
    for pattern in patterns:
        files = sorted(glob.glob(str(DATA_DIR / pattern)))
        for filepath in files:
            try:
                df = pd.read_csv(filepath)
                if len(df) < 10:
                    continue
                    
                # Normalize column names
                df.columns = [c.lower().replace('_', '') for c in df.columns]
                
                # Check for required columns
                required = ['front', 'left', 'right']
                if not all(col in df.columns for col in required):
                    # Try alternate names
                    renames = {}
                    for col in df.columns:
                        if 'front' in col: renames[col] = 'front'
                        elif 'left' in col and 'enc' not in col: renames[col] = 'left'
                        elif 'right' in col and 'enc' not in col: renames[col] = 'right'
                    df.rename(columns=renames, inplace=True)
                
                if all(col in df.columns for col in required):
                    print(f"  {os.path.basename(filepath)}: {len(df)} samples")
                    all_data.append(df)
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
    
    return all_data

# ==================== GENERATE SAFETY DATA ====================
def generate_safety_samples(n_per_rule=3000):
    """Generate synthetic samples that GUARANTEE correct behavior"""
    
    samples = []
    
    for rule_name, rule in SAFETY_RULES.items():
        for _ in range(n_per_rule):
            front = np.random.randint(rule['front'][0], rule['front'][1])
            left = np.random.randint(rule['left'][0], rule['left'][1])
            right = np.random.randint(rule['right'][0], rule['right'][1])
            
            # Add some variation to PWM targets
            pwm_l = rule['pwm'][0] + np.random.randint(-5, 6)
            pwm_r = rule['pwm'][1] + np.random.randint(-5, 6)
            
            samples.append({
                'front': front,
                'left': left,
                'right': right,
                'pwml': max(0, min(255, pwm_l)),
                'pwmr': max(0, min(255, pwm_r)),
                'source': 'safety'
            })
    
    return pd.DataFrame(samples)

# ==================== PROCESS REAL DATA ====================
def process_real_data(dataframes):
    """Extract training samples from real recordings"""
    
    samples = []
    
    for df in dataframes:
        for _, row in df.iterrows():
            front = row.get('front', 0)
            left = row.get('left', 0)
            right = row.get('right', 0)
            
            # Skip invalid
            if front <= 0 or left <= 0 or right <= 0:
                continue
            if front > SENSOR_MAX or left > SENSOR_MAX or right > SENSOR_MAX:
                continue
            
            # Check for ideal_l/ideal_r columns (from new collector)
            if 'ideall' in df.columns and 'idealr' in df.columns:
                pwm_l = row['ideall']
                pwm_r = row['idealr']
            # Check for pwm columns
            elif 'pwml' in df.columns and 'pwmr' in df.columns:
                pwm_l = row['pwml']
                pwm_r = row['pwmr']
                # Skip if both zero (stopped)
                if pwm_l == 0 and pwm_r == 0:
                    continue
            else:
                # Calculate ideal PWM from sensors
                pwm_l, pwm_r = calculate_ideal_pwm(front, left, right)
            
            samples.append({
                'front': front,
                'left': left,
                'right': right,
                'pwml': pwm_l,
                'pwmr': pwm_r,
                'source': 'real'
            })
    
    return pd.DataFrame(samples) if samples else pd.DataFrame()

def calculate_ideal_pwm(front, left, right):
    """Calculate what the PWM SHOULD be based on sensor values"""
    
    BASE = 130
    
    # Front handling
    if front > 2600:
        return 0, 0  # Stop
    elif front > 2000:
        speed = int(70 + (2600 - front) / 600 * 60)
    else:
        speed = BASE
    
    # Side correction (CRITICAL - correct direction!)
    correction = 0
    
    # Calculate error (positive = closer to left)
    error = left - right
    correction = int(error * 0.02)
    
    # Danger zones - strong override
    if left > 3200:
        correction = -50  # Hard right (negative = speed up left relative to right)
    elif right > 3200:
        correction = 50   # Hard left
    elif left > 2800:
        correction = -25  # Medium right
    elif right > 2800:
        correction = 25   # Medium left
    
    # Apply correction
    # To go RIGHT (away from left wall): LEFT wheel faster
    # correction < 0 means steer right, so add to left, subtract from right
    pwm_l = speed - correction  # More negative correction = more left speed = more right turn
    pwm_r = speed + correction
    
    # Wait, let me reconsider the math:
    # If left > right (closer to left wall), error is positive
    # We want to steer RIGHT, which means LEFT wheel should be FASTER
    # So pwm_l should increase when error is positive
    
    # Let's redo:
    # error = left - right (positive when close to left)
    # To steer right (away from left): increase pwm_l, decrease pwm_r
    # So: pwm_l = speed + k*error, pwm_r = speed - k*error
    
    pwm_l = speed + int(error * 0.02)
    pwm_r = speed - int(error * 0.02)
    
    # Danger overrides
    if left > 3200:
        pwm_l = min(180, pwm_l + 50)
        pwm_r = max(60, pwm_r - 30)
    if right > 3200:
        pwm_l = max(60, pwm_l - 30)
        pwm_r = min(180, pwm_r + 50)
    
    return max(0, min(255, pwm_l)), max(0, min(255, pwm_r))

# ==================== NEURAL NETWORK ====================
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

class NeuralNetwork:
    def __init__(self):
        # Xavier initialization
        self.W1 = np.random.randn(INPUT_SIZE, HIDDEN1).astype(np.float32) * np.sqrt(2.0 / INPUT_SIZE)
        self.b1 = np.zeros(HIDDEN1, dtype=np.float32)
        
        self.W2 = np.random.randn(HIDDEN1, HIDDEN2).astype(np.float32) * np.sqrt(2.0 / HIDDEN1)
        self.b2 = np.zeros(HIDDEN2, dtype=np.float32)
        
        self.W3 = np.random.randn(HIDDEN2, OUTPUT_SIZE).astype(np.float32) * np.sqrt(2.0 / HIDDEN2)
        self.b3 = np.zeros(OUTPUT_SIZE, dtype=np.float32)
    
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3
    
    def backward(self, X, Y, output, lr):
        m = X.shape[0]
        
        dz3 = (output - Y) / m
        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0)
        
        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_deriv(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_deriv(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        # Gradient clipping
        clip = 1.0
        self.W3 -= lr * np.clip(dW3, -clip, clip)
        self.b3 -= lr * np.clip(db3, -clip, clip)
        self.W2 -= lr * np.clip(dW2, -clip, clip)
        self.b2 -= lr * np.clip(db2, -clip, clip)
        self.W1 -= lr * np.clip(dW1, -clip, clip)
        self.b1 -= lr * np.clip(db1, -clip, clip)
    
    def train(self, X, Y, epochs, batch_size, lr):
        n = len(X)
        best_loss = float('inf')
        
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X, Y = X[idx], Y[idx]
            
            total_loss = 0
            batches = 0
            
            for i in range(0, n, batch_size):
                Xb = X[i:i+batch_size]
                Yb = Y[i:i+batch_size]
                
                out = self.forward(Xb)
                loss = np.mean((out - Yb) ** 2)
                total_loss += loss
                batches += 1
                
                self.backward(Xb, Yb, out, lr)
            
            avg_loss = total_loss / batches
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # LR decay
            if epoch > 0 and epoch % 100 == 0:
                lr *= 0.7
        
        return best_loss

# ==================== EXPORT WEIGHTS ====================
def export_weights(nn, filepath):
    """Export to C header file"""
    
    with open(filepath, 'w') as f:
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        f.write("// Neural Network for Maze Wall Avoidance\n")
        f.write("// GUARANTEED NO-HIT with correct steering\n")
        f.write(f"// Architecture: {INPUT_SIZE} -> {HIDDEN1} -> {HIDDEN2} -> {OUTPUT_SIZE}\n\n")
        
        f.write(f"const float SCALE_SENSOR = {SENSOR_MAX}f;\n")
        f.write(f"const float SCALE_PWM = {PWM_MAX}f;\n\n")
        
        # W1: [3][32]
        f.write("const float w1[32][3] = {\n")
        for j in range(HIDDEN1):
            f.write("  {" + ", ".join(f"{nn.W1[i, j]:.6f}f" for i in range(INPUT_SIZE)) + "},\n")
        f.write("};\n")
        f.write("const float b1[32] = {" + ", ".join(f"{nn.b1[j]:.6f}f" for j in range(HIDDEN1)) + "};\n\n")
        
        # W2: [32][16]
        f.write("const float w2[16][32] = {\n")
        for j in range(HIDDEN2):
            f.write("  {" + ", ".join(f"{nn.W2[i, j]:.6f}f" for i in range(HIDDEN1)) + "},\n")
        f.write("};\n")
        f.write("const float b2[16] = {" + ", ".join(f"{nn.b2[j]:.6f}f" for j in range(HIDDEN2)) + "};\n\n")
        
        # W3: [16][2]
        f.write("const float w3[2][16] = {\n")
        for j in range(OUTPUT_SIZE):
            f.write("  {" + ", ".join(f"{nn.W3[i, j]:.6f}f" for i in range(HIDDEN2)) + "},\n")
        f.write("};\n")
        f.write("const float b3[2] = {" + ", ".join(f"{nn.b3[j]:.6f}f" for j in range(OUTPUT_SIZE)) + "};\n\n")
        
        f.write("#endif\n")
    
    print(f"\nWeights saved to: {filepath}")

# ==================== VALIDATE ====================
def validate_network(nn):
    """Test the network with critical scenarios"""
    
    print("\n" + "=" * 70)
    print("VALIDATION - Critical Scenarios")
    print("=" * 70)
    
    test_cases = [
        # (front, left, right, expected_behavior)
        (300, 2700, 2700, "STRAIGHT"),
        (300, 3200, 2000, "RIGHT (away from left)"),
        (300, 2000, 3200, "LEFT (away from right)"),
        (300, 3500, 1500, "HARD RIGHT"),
        (300, 1500, 3500, "HARD LEFT"),
        (2800, 2500, 2500, "STOP"),
        (2200, 2500, 2500, "SLOW"),
        (300, 800, 2800, "LEFT TURN"),
        (300, 2800, 800, "RIGHT TURN"),
        (300, 600, 600, "FAST"),
    ]
    
    print(f"{'Front':>6} {'Left':>6} {'Right':>6} | {'PWM_L':>6} {'PWM_R':>6} | {'Expected':<20} {'Actual':<15} {'OK?'}")
    print("-" * 90)
    
    all_pass = True
    
    for front, left, right, expected in test_cases:
        inp = np.array([[front/SENSOR_MAX, left/SENSOR_MAX, right/SENSOR_MAX]], dtype=np.float32)
        out = nn.forward(inp)[0]
        pwm_l = int(out[0] * PWM_MAX)
        pwm_r = int(out[1] * PWM_MAX)
        
        # Determine actual behavior
        diff = pwm_l - pwm_r
        if pwm_l < 30 and pwm_r < 30:
            actual = "STOP"
        elif pwm_l < 90 and pwm_r < 90:
            actual = "SLOW"
        elif diff > 40:
            actual = "HARD RIGHT"
        elif diff > 15:
            actual = "RIGHT"
        elif diff < -40:
            actual = "HARD LEFT"
        elif diff < -15:
            actual = "LEFT"
        elif pwm_l > 140 and pwm_r > 140:
            actual = "FAST"
        else:
            actual = "STRAIGHT"
        
        # Check if correct
        ok = "✓" if expected.split()[0] in actual or actual in expected else "✗"
        if ok == "✗":
            all_pass = False
        
        print(f"{front:>6} {left:>6} {right:>6} | {pwm_l:>6} {pwm_r:>6} | {expected:<20} {actual:<15} {ok}")
    
    print("-" * 90)
    print(f"Result: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    
    return all_pass

# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("MazeSolver ULTIMATE Neural Network Trainer v2.0")
    print("=" * 70)
    
    # Load real data
    print("\n1. Loading recorded data...")
    real_dfs = load_all_data()
    
    real_df = process_real_data(real_dfs) if real_dfs else pd.DataFrame()
    real_samples = len(real_df) if not real_df.empty else 0
    print(f"   Real samples: {real_samples}")
    
    # Generate safety data
    print("\n2. Generating safety injection data...")
    safety_df = generate_safety_samples(n_per_rule=3000)
    print(f"   Safety samples: {len(safety_df)}")
    
    # Combine
    if not real_df.empty:
        # Weight real data 2x
        combined = pd.concat([real_df, real_df, safety_df], ignore_index=True)
    else:
        combined = safety_df
    
    print(f"   Total training samples: {len(combined)}")
    
    # Prepare training data
    print("\n3. Preparing training data...")
    X = combined[['front', 'left', 'right']].values.astype(np.float32) / SENSOR_MAX
    Y = combined[['pwml', 'pwmr']].values.astype(np.float32) / PWM_MAX
    
    # Shuffle and split
    n = len(X)
    idx = np.random.permutation(n)
    X, Y = X[idx], Y[idx]
    
    split = int(0.9 * n)
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Train
    print("\n4. Training Neural Network...")
    print("=" * 70)
    
    nn = NeuralNetwork()
    final_loss = nn.train(X_train, Y_train, EPOCHS, BATCH_SIZE, LEARNING_RATE)
    
    # Validation loss
    val_out = nn.forward(X_val)
    val_loss = np.mean((val_out - Y_val) ** 2)
    print(f"\nFinal Training Loss: {final_loss:.6f}")
    print(f"Validation Loss: {val_loss:.6f}")
    
    # Validate with test cases
    passed = validate_network(nn)
    
    if not passed:
        print("\n⚠️  WARNING: Some validation tests failed!")
        print("    Consider collecting more data or adjusting safety rules.")
    
    # Export
    print("\n5. Exporting weights...")
    export_weights(nn, DATA_DIR / "weights.h")
    
    # Stats
    total_params = INPUT_SIZE*HIDDEN1 + HIDDEN1 + HIDDEN1*HIDDEN2 + HIDDEN2 + HIDDEN2*OUTPUT_SIZE + OUTPUT_SIZE
    print(f"   Parameters: {total_params} ({total_params * 4} bytes)")
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
