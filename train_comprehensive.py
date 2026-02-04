#!/usr/bin/env python3
"""
MazeSolver Comprehensive Training Script
=========================================
Uses ALL available data (train*.csv, run*.csv, processed_train_data.csv)
to train a robust neural network for maze solving.

Strategy:
1. Load all CSV files and extract sensor->action mappings
2. Filter out bad samples (crashes, stuck states)
3. Inject safety rules (wall avoidance)
4. Train neural network with data augmentation
5. Export optimized weights.h for Teensy
"""

import numpy as np
import pandas as pd
import glob
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
SENSOR_MAX = 4095.0  # 12-bit ADC
PWM_MAX = 255.0
BASE_SPEED = 150
TURN_SPEED_DIFF = 80

# Neural Network Architecture
HIDDEN_LAYERS = (32, 16)
ACTIVATION = 'tanh'
MAX_ITERATIONS = 5000
LEARNING_RATE = 0.001

# Safety thresholds (from your robot calibration)
SIDE_DANGER = 3200   # Too close to wall
SIDE_WALL = 1800     # Wall detected
FRONT_STOP = 2500    # Must stop

print("=" * 60)
print("MazeSolver Neural Network Training")
print("=" * 60)

# ==================== DATA LOADING ====================
def load_all_data():
    """Load and combine all CSV files"""
    all_data = []
    
    # Load train*.csv files (basic sensor + encoder data)
    train_files = glob.glob('train*.csv')
    print(f"\n📁 Found {len(train_files)} training files")
    for file in train_files:
        try:
            df = pd.read_csv(file)
            if 'Front' in df.columns and 'Left' in df.columns and 'Right' in df.columns:
                # Calculate velocity from encoder deltas
                if 'EncL' in df.columns and 'EncR' in df.columns:
                    df['vL'] = df['EncL'].diff().fillna(0)
                    df['vR'] = df['EncR'].diff().fillna(0)
                    # Convert to PWM estimate
                    df['pwm_l'] = np.clip(df['vL'] * 20, 0, PWM_MAX)
                    df['pwm_r'] = np.clip(df['vR'] * 20, 0, PWM_MAX)
                    all_data.append(df[['Front', 'Left', 'Right', 'pwm_l', 'pwm_r']])
                    print(f"  ✓ {file}: {len(df)} samples")
        except Exception as e:
            print(f"  ✗ {file}: {e}")
    
    # Load run*.csv files (detailed logs with PWM values)
    run_files = glob.glob('run*.csv')
    print(f"\n📁 Found {len(run_files)} run log files")
    for file in run_files:
        try:
            df = pd.read_csv(file)
            if 'f' in df.columns:  # run_latest.csv format
                df = df.rename(columns={'f': 'Front', 'l': 'Left', 'r': 'Right', 
                                       'pwmL': 'pwm_l', 'pwmR': 'pwm_r'})
            
            if all(col in df.columns for col in ['Front', 'Left', 'Right', 'pwm_l', 'pwm_r']):
                # Filter out stopped states and crashes
                df = df[(df['pwm_l'] > 0) & (df['pwm_r'] > 0)]
                df = df[df['Front'] < FRONT_STOP]  # Only moving forward samples
                all_data.append(df[['Front', 'Left', 'Right', 'pwm_l', 'pwm_r']])
                print(f"  ✓ {file}: {len(df)} samples")
        except Exception as e:
            print(f"  ✗ {file}: {e}")
    
    # Load processed_train_data.csv
    if glob.glob('processed_train_data.csv'):
        try:
            df = pd.read_csv('processed_train_data.csv')
            if 'vL_smooth' in df.columns and 'vR_smooth' in df.columns:
                df['pwm_l'] = np.clip(df['vL_smooth'] / 2, 0, PWM_MAX)
                df['pwm_r'] = np.clip(df['vR_smooth'] / 2, 0, PWM_MAX)
                all_data.append(df[['Front', 'Left', 'Right', 'pwm_l', 'pwm_r']])
                print(f"  ✓ processed_train_data.csv: {len(df)} samples")
        except Exception as e:
            print(f"  ✗ processed_train_data.csv: {e}")
    
    if not all_data:
        raise ValueError("No valid data files found!")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total raw samples: {len(combined)}")
    return combined

# ==================== DATA CLEANING ====================
def clean_data(df):
    """Remove bad samples and outliers"""
    print("\n🧹 Cleaning data...")
    original_len = len(df)
    
    # Remove stopped states
    df = df[(df['pwm_l'].abs() > 10) | (df['pwm_r'].abs() > 10)]
    
    # Remove sensor errors (0 values usually mean disconnection)
    df = df[(df['Front'] > 0) & (df['Left'] > 0) & (df['Right'] > 0)]
    
    # Remove extreme outliers
    df = df[df['Front'] < SENSOR_MAX]
    df = df[df['Left'] < SENSOR_MAX]
    df = df[df['Right'] < SENSOR_MAX]
    df = df[df['pwm_l'] <= PWM_MAX]
    df = df[df['pwm_r'] <= PWM_MAX]
    
    print(f"  Removed {original_len - len(df)} bad samples ({len(df)} remaining)")
    return df

# ==================== SAFETY INJECTION ====================
def inject_safety_samples(n_samples=3000):
    """Generate synthetic samples to ensure wall avoidance"""
    print(f"\n🛡️  Injecting {n_samples} safety samples...")
    safety = []
    
    for _ in range(n_samples // 3):
        # Rule 1: Left wall too close -> steer right
        safety.append({
            'Front': np.random.randint(300, 1800),
            'Left': np.random.randint(SIDE_DANGER, SENSOR_MAX),
            'Right': np.random.randint(500, SIDE_WALL),
            'pwm_l': BASE_SPEED + TURN_SPEED_DIFF,
            'pwm_r': BASE_SPEED - TURN_SPEED_DIFF
        })
        
        # Rule 2: Right wall too close -> steer left
        safety.append({
            'Front': np.random.randint(300, 1800),
            'Left': np.random.randint(500, SIDE_WALL),
            'Right': np.random.randint(SIDE_DANGER, SENSOR_MAX),
            'pwm_l': BASE_SPEED - TURN_SPEED_DIFF,
            'pwm_r': BASE_SPEED + TURN_SPEED_DIFF
        })
        
        # Rule 3: Centered in corridor -> go straight
        safety.append({
            'Front': np.random.randint(300, 1800),
            'Left': np.random.randint(SIDE_WALL, SIDE_DANGER),
            'Right': np.random.randint(SIDE_WALL, SIDE_DANGER),
            'pwm_l': BASE_SPEED,
            'pwm_r': BASE_SPEED
        })
    
    return pd.DataFrame(safety)

# ==================== DATA AUGMENTATION ====================
def augment_data(df):
    """Mirror data to double training samples"""
    print("\n🔄 Augmenting data (mirroring)...")
    df_mirror = df.copy()
    df_mirror['Left'], df_mirror['Right'] = df['Right'].copy(), df['Left'].copy()
    df_mirror['pwm_l'], df_mirror['pwm_r'] = df['pwm_r'].copy(), df['pwm_l'].copy()
    
    combined = pd.concat([df, df_mirror], ignore_index=True)
    print(f"  Augmented to {len(combined)} samples")
    return combined

# ==================== TRAINING ====================
def train_neural_network(df):
    """Train the neural network"""
    print("\n🧠 Training Neural Network...")
    
    # Prepare features and targets
    X = df[['Front', 'Left', 'Right']].values / SENSOR_MAX  # Normalize to [0, 1]
    y = df[['pwm_l', 'pwm_r']].values / PWM_MAX  # Normalize to [0, 1]
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Architecture: 3 -> {HIDDEN_LAYERS[0]} -> {HIDDEN_LAYERS[1]} -> 2")
    
    # Create and train model
    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation=ACTIVATION,
        solver='adam',
        learning_rate_init=LEARNING_RATE,
        max_iter=MAX_ITERATIONS,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    print(f"\n  ✓ Training R² score: {train_score:.4f}")
    print(f"  ✓ Validation R² score: {val_score:.4f}")
    
    return model

# ==================== VALIDATION ====================
def validate_model(model):
    """Test model on critical scenarios"""
    print("\n🧪 Validation Tests:")
    
    test_cases = [
        ([1000, 3800, 1500], "LEFT WALL DANGER"),
        ([1000, 1500, 3800], "RIGHT WALL DANGER"),
        ([1000, 2200, 2200], "CENTERED CORRIDOR"),
        ([3000, 2000, 2000], "FRONT WALL"),
        ([500, 1000, 1000], "OPEN SPACE")
    ]
    
    for sensors, description in test_cases:
        X_test = np.array([[sensors[0]/SENSOR_MAX, sensors[1]/SENSOR_MAX, sensors[2]/SENSOR_MAX]])
        y_pred = model.predict(X_test)[0]
        pwm_l = int(y_pred[0] * PWM_MAX)
        pwm_r = int(y_pred[1] * PWM_MAX)
        
        print(f"  {description:20s} -> L:{pwm_l:3d} R:{pwm_r:3d}", end="")
        
        # Validate response
        if "LEFT" in description and pwm_l > pwm_r:
            print(" ✓ (steering right)")
        elif "RIGHT" in description and pwm_l < pwm_r:
            print(" ✓ (steering left)")
        elif "CENTERED" in description and abs(pwm_l - pwm_r) < 20:
            print(" ✓ (going straight)")
        elif "FRONT" in description and (pwm_l < 100 or pwm_r < 100):
            print(" ✓ (slowing down)")
        else:
            print(" ⚠️  (unexpected)")

# ==================== EXPORT WEIGHTS ====================
def export_weights(model, filename='weights.h'):
    """Export trained weights to C header file"""
    print(f"\n📝 Exporting weights to {filename}...")
    
    w1 = model.coefs_[0]  # [3, 32]
    b1 = model.intercepts_[0]  # [32]
    w2 = model.coefs_[1]  # [32, 16]
    b2 = model.intercepts_[1]  # [16]
    w3 = model.coefs_[2]  # [16, 2]
    b3 = model.intercepts_[2]  # [2]
    
    with open(filename, 'w') as f:
        f.write("// Auto-generated Neural Network Weights\n")
        f.write("// Trained on comprehensive maze-solving data\n")
        f.write(f"// Architecture: 3 -> {HIDDEN_LAYERS[0]} -> {HIDDEN_LAYERS[1]} -> 2\n")
        f.write(f"// Activation: {ACTIVATION}\n\n")
        
        # Constants
        f.write(f"#define SENSOR_MAX {SENSOR_MAX:.1f}f\n")
        f.write(f"#define PWM_MAX {PWM_MAX:.1f}f\n\n")
        
        # Layer 1: Input -> Hidden1 (3 -> 32)
        f.write(f"const float w1[{w1.shape[1]}][{w1.shape[0]}] = {{\n")
        for i in range(w1.shape[1]):
            f.write("  {")
            f.write(", ".join([f"{w1[j, i]:.6f}f" for j in range(w1.shape[0])]))
            f.write("},\n")
        f.write("};\n\n")
        
        f.write(f"const float b1[{len(b1)}] = {{\n")
        f.write("  " + ", ".join([f"{x:.6f}f" for x in b1]) + "\n")
        f.write("};\n\n")
        
        # Layer 2: Hidden1 -> Hidden2 (32 -> 16)
        f.write(f"const float w2[{w2.shape[1]}][{w2.shape[0]}] = {{\n")
        for i in range(w2.shape[1]):
            f.write("  {")
            f.write(", ".join([f"{w2[j, i]:.6f}f" for j in range(w2.shape[0])]))
            f.write("},\n")
        f.write("};\n\n")
        
        f.write(f"const float b2[{len(b2)}] = {{\n")
        f.write("  " + ", ".join([f"{x:.6f}f" for x in b2]) + "\n")
        f.write("};\n\n")
        
        # Layer 3: Hidden2 -> Output (16 -> 2)
        f.write(f"const float w3[{w3.shape[1]}][{w3.shape[0]}] = {{\n")
        for i in range(w3.shape[1]):
            f.write("  {")
            f.write(", ".join([f"{w3[j, i]:.6f}f" for j in range(w3.shape[0])]))
            f.write("},\n")
        f.write("};\n\n")
        
        f.write(f"const float b3[{len(b3)}] = {{\n")
        f.write("  " + ", ".join([f"{x:.6f}f" for x in b3]) + "\n")
        f.write("};\n")
    
    print(f"  ✓ Exported {filename} successfully!")

# ==================== MAIN ====================
def main():
    try:
        # Load all data
        df = load_all_data()
        
        # Clean data
        df = clean_data(df)
        
        # Inject safety samples
        safety_df = inject_safety_samples(3000)
        df = pd.concat([df, safety_df], ignore_index=True)
        
        # Augment data
        df = augment_data(df)
        
        print(f"\n📊 Final dataset: {len(df)} samples")
        
        # Train model
        model = train_neural_network(df)
        
        # Validate
        validate_model(model)
        
        # Export weights
        export_weights(model, 'weights.h')
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("Next steps:")
        print("  1. Upload bot_nn.ino to your Teensy")
        print("  2. Test in the maze")
        print("  3. Collect more data if needed")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
