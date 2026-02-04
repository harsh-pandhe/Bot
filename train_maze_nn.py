import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import glob

# 1. Load all your recorded files
files = glob.glob('train*.csv') + glob.glob('run*.csv')
print(f"Loading {len(files)} data files...")

li = []
for f in files:
    try:
        df = pd.read_csv(f)
        if df.empty: continue
        # Normalize column names for different versions
        df.columns = [c.capitalize() if c.islower() else c for c in df.columns]
        if 'Time_ms' in df.columns: df.rename(columns={'Time_ms': 'Millis'}, inplace=True)
        
        # Calculate Target Velocities from Encoders
        df['dL'] = df['Encl'].diff().fillna(0)
        df['dR'] = df['Encr'].diff().fillna(0)
        df['dt'] = df['Millis'].diff().fillna(40).replace(0, 40)
        df['vL'] = (df['dL'] / (df['dt'] / 1000.0)).rolling(window=7).mean().fillna(0)
        df['vR'] = (df['dR'] / (df['dt'] / 1000.0)).rolling(window=7).mean().fillna(0)
        
        li.append(df[['Front', 'Left', 'Right', 'vL', 'vR']])
    except: continue

main_df = pd.concat(li, ignore_index=True)
main_df = main_df[main_df[['vL', 'vR']].abs().sum(axis=1) > 30] # Filter noise

# 2. 🔥 SAFETY INJECTION: Force the Brain to learn "NO-HIT" logic
# We add fake samples where the sensors are near walls to teach hard recovery.
safety = []
for _ in range(2000):
    # If Left is very close (>3200), steer hard Right
    safety.append({'Front': np.random.randint(200, 1000), 'Left': np.random.randint(3200, 4000), 
                   'Right': np.random.randint(500, 1500), 'vL': 2500, 'vR': -800})
    # If Right is very close (>3200), steer hard Left
    safety.append({'Front': np.random.randint(200, 1000), 'Left': np.random.randint(500, 1500), 
                   'Right': np.random.randint(3200, 4000), 'vL': -800, 'vR': 2500})
    # If Front is blocked (>3000), stop
    safety.append({'Front': np.random.randint(3000, 4095), 'Left': np.random.randint(1000, 3000), 
                   'Right': np.random.randint(1000, 3000), 'vL': 0, 'vR': 0})

df_final = pd.concat([main_df, pd.DataFrame(safety)], ignore_index=True)

# 3. Mirroring & Scaling
X = df_final[['Front', 'Left', 'Right']].values / 4095.0
target_scaler = MinMaxScaler(feature_range=(-1, 1))
y = target_scaler.fit_transform(df_final[['vL', 'vR']].values)

# 4. Train the 32x16 Deep Brain
model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='tanh', solver='adam', max_iter=5000)
model.fit(X, y)
print(f"Brain Training Complete. Confidence: {model.score(X, y):.2f}")

# 5. Export weights.h (Copy output to your Teensy project)
def export_weights(model, sl, sr):
    with open('weights.h', 'w') as f:
        f.write(f"const float SCALE_L = {sl};\nconst float SCALE_R = {sr};\n")
        f.write(f"const float w1[32][3] = {{")
        for r in model.coefs_[0].T: f.write(" {" + ",".join(map(str, np.round(r, 6))) + "},")
        f.write("}};\nconst float b1[32] = {{" + ",".join(map(str, np.round(model.intercepts_[0], 6))) + "}};\n")
        # Repeat for w2, b2, w3, b3...
export_weights(model, target_scaler.data_max_[0], target_scaler.data_max_[1])