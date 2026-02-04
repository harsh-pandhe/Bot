import pandas as pd
import numpy as np
import glob
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

# 1. Load and Combine All Data
files = glob.glob("train*.csv")
print(f"Found files: {files}")

li = []
for f in files:
    try:
        temp_df = pd.read_csv(f)
        # Calculate raw velocities for this specific run
        temp_df['dEncL'] = temp_df['EncL'].diff().fillna(0)
        temp_df['dEncR'] = temp_df['EncR'].diff().fillna(0)
        temp_df['dMillis'] = temp_df['Millis'].diff().fillna(40).replace(0, 40)
        
        # Velocity = ticks / seconds
        temp_df['vL'] = temp_df['dEncL'] / (temp_df['dMillis'] / 1000.0)
        temp_df['vR'] = temp_df['dEncR'] / (temp_df['dMillis'] / 1000.0)
        
        # Smooth to remove hand-shake noise
        temp_df['vL_smooth'] = temp_df['vL'].rolling(window=5).mean().fillna(0)
        temp_df['vR_smooth'] = temp_df['vR'].rolling(window=5).mean().fillna(0)
        
        li.append(temp_df)
    except:
        print(f"Skipping empty or broken file: {f}")

df = pd.concat(li, ignore_index=True)

# 2. Data Cleaning
# Remove rows where the bot wasn't moving (stops it from learning to "get stuck")
df = df[df[['vL_smooth', 'vR_smooth']].abs().sum(axis=1) > 50]

# 3. Data Augmentation (Mirroring)
# Flips left/right sensors and motors to double turn-data accuracy
df_mirrored = df.copy()
df_mirrored['Left'], df_mirrored['Right'] = df['Right'], df['Left']
df_mirrored['vL_smooth'], df_mirrored['vR_smooth'] = df['vR_smooth'], df['vL_smooth']
df = pd.concat([df, df_mirrored], ignore_index=True)

# 4. Scaling
X = df[['Front', 'Left', 'Right']].values / 4095.0 # Normalize ADC
target_scaler = MinMaxScaler(feature_range=(-1, 1))
y = target_scaler.fit_transform(df[['vL_smooth', 'vR_smooth']].values) # Scale to -1, 1

MAX_SPEED_L = target_scaler.data_max_[0]
MAX_SPEED_R = target_scaler.data_max_[1]

print(f"Training on {len(X)} samples...")

# 5. Deep Neural Network Training
model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation='tanh',
    solver='adam',
    max_iter=10000,
    tol=1e-5,
    random_state=1
)

model.fit(X, y)
print("Training Complete! New Score:", model.score(X, y))

# 6. Export to C++ weights.h
def export_weights(model, sl, sr):
    w1 = model.coefs_[0].T
    b1 = model.intercepts_[0]
    w2 = model.coefs_[1].T
    b2 = model.intercepts_[1]
    w3 = model.coefs_[2].T
    b3 = model.intercepts_[2]
    
    with open('weights.h', 'w') as f:
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        f.write(f"const float SCALE_L = {sl};\nconst float SCALE_R = {sr};\n\n")
        
        # Layer 1
        f.write("const float w1[32][3] = {\n")
        for row in w1: f.write("  {" + ", ".join([str(round(x, 6)) for x in row]) + "},\n")
        f.write("};\n")
        f.write("const float b1[32] = {" + ", ".join([str(round(x, 6)) for x in b1]) + "};\n\n")
        
        # Layer 2
        f.write("const float w2[16][32] = {\n")
        for row in w2: f.write("  {" + ", ".join([str(round(x, 6)) for x in row]) + "},\n")
        f.write("};\n")
        f.write("const float b2[16] = {" + ", ".join([str(round(x, 6)) for x in b2]) + "};\n\n")
        
        # Layer 3
        f.write("const float w3[2][16] = {\n")
        for row in w3: f.write("  {" + ", ".join([str(round(x, 6)) for x in row]) + "},\n")
        f.write("};\n")
        f.write("const float b3[2] = {" + ", ".join([str(round(x, 6)) for x in b3]) + "};\n\n")
        f.write("#endif")

export_weights(model, MAX_SPEED_L, MAX_SPEED_R)
print("Success! 'weights.h' is ready for Teensy.")