# MazeSolver Robot - Neural Network Based Navigation

An Arduino-based autonomous maze-solving robot that uses neural networks for intelligent navigation, combining sensor data with machine learning to navigate complex environments.

## 🤖 Overview

This project implements an autonomous robot capable of navigating mazes using:
- **Neural Network Control**: ML-based decision making for motor control
- **Multi-Sensor Fusion**: Front, left, and right proximity sensors
- **Encoder-based Navigation**: Precise positioning and turning
- **Real-time Learning**: Data collection for continuous improvement
- **OLED Display**: Live status and debugging information

## 🔧 Hardware Requirements

- **Microcontroller**: Teensy (or compatible Arduino board)
- **Motor Driver**: TB6612FNG or similar
- **Motors**: DC motors with encoders
- **Sensors**: 3x analog proximity sensors (front, left, right)
  - ADC: 12-bit resolution (0-4095)
- **Display**: Adafruit SH110X OLED (128x64)
- **Storage**: SD card for data logging
- **Power**: Appropriate battery pack for motors and logic

## 📁 Project Structure

### Arduino Firmware
- `bot_ultimate.ino` - Main robot firmware with neural network integration
- `bot_maze_nn_hybrid.ino` - Hybrid control (NN + rule-based)
- `bot_maze_solver.ino` - Pure maze-solving algorithm
- `bot_v*.ino` - Various development versions
- `data_collector_v*.ino` - Data collection firmware
- `calibrate_turn.ino` - Encoder calibration utility
- `weights.h` - Neural network weights (auto-generated)

### Python Training Scripts
- `train_comprehensive.py` - **Main training script** using all available data
- `train_final.py` - Optimized training pipeline
- `train_ultimate.py` - Advanced training with safety rules
- `train_nn.py` - Basic neural network trainer
- `train_bot.py` - Original training script

### Data
- `data/` - Training data collected from robot runs
  - `run_*.csv` - Individual run data
  - `train*.csv` - Labeled training datasets
  - `processed_train_data.csv` - Preprocessed data
  - `run **/` - Organized run sessions

## 🚀 Quick Start

### 1. Hardware Setup
1. Wire all components according to pin definitions in firmware
2. Calibrate sensors and encoder ticks (see Configuration section)
3. Upload data collection firmware to gather training data

### 2. Data Collection
```bash
# Upload data_collector_v3.ino to your robot
# Run the robot in your maze
# Copy CSV files from SD card to data/ folder
```

### 3. Train Neural Network
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_comprehensive.py

# This will generate weights.h file
```

### 4. Deploy
```bash
# Upload bot_ultimate.ino with the new weights.h
# The robot will now use the trained neural network
```

## ⚙️ Configuration

### Sensor Thresholds (in firmware)
```cpp
const int16_t FRONT_EMERGENCY = 2700;  // Hard stop
const int16_t FRONT_CLOSE = 2200;      // Start slowing
const int16_t SIDE_CRASH = 3200;       // Emergency avoidance
const int16_t SIDE_DANGER = 2900;      // Strong correction
const int16_t SIDE_WALL = 2000;        // Wall detected
```

### Motor Parameters
```cpp
const int16_t MAX_SPEED = 160;
const int16_t CRUISE_SPEED = 130;
const int16_t TURN_SPEED = 120;
const long TICKS_90 = 455;  // Adjust for your robot
```

### Neural Network Architecture
```python
HIDDEN_LAYERS = (32, 16)  # Two hidden layers
ACTIVATION = 'tanh'
MAX_ITERATIONS = 5000
LEARNING_RATE = 0.001
```

## 🧠 How It Works

### Neural Network Training
1. **Data Collection**: Robot runs in manual/semi-auto mode, logging:
   - Sensor readings (front, left, right)
   - Motor PWM values
   - Encoder positions
   - Timestamps

2. **Data Processing**:
   - Normalize sensor values (0-4095 → 0-1)
   - Filter bad samples (crashes, stuck states)
   - Inject safety rules for wall avoidance
   - Data augmentation for better generalization

3. **Training**:
   - Input: 3 normalized sensor values
   - Output: 2 PWM values (left motor, right motor)
   - Architecture: Multi-layer perceptron with tanh activation
   - Export: Quantized weights in C header format

4. **Inference**: 
   - Real-time prediction on Teensy (< 1ms per inference)
   - Safety overrides prevent crashes
   - Fallback to rule-based control if needed

### Control Strategy
```
Sensors → Neural Network → Motor PWM
    ↓
Safety Layer → Hard limits → Final Output
```

## 📊 Data Format

CSV files contain:
```
timestamp,sensorF,sensorL,sensorR,pwmL,pwmR,encL,encR
1234,2000,1500,1600,150,150,100,98
```

## 🛠️ Development

### Adding New Features
1. Modify firmware in `*.ino` files
2. Collect new training data
3. Retrain neural network
4. Test incrementally

### Debugging
- OLED displays real-time sensor and motor values
- SD card logging for post-analysis
- Serial monitor for detailed debugging

### Calibration
```bash
# Upload calibrate_turn.ino
# Measure encoder ticks for 90° turn
# Update TICKS_90 in firmware
```

## 📈 Performance

- **Response Time**: < 1ms neural network inference
- **Turn Accuracy**: ±2° with encoder feedback
- **Wall Detection**: Reliable at 5-30cm range
- **Success Rate**: Depends on training data quality

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Advanced path planning algorithms
- Multi-maze generalization
- Real-time learning/adaptation
- Vision-based navigation
- ROS integration

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Arduino/Teensy community
- scikit-learn for ML tools
- Adafruit libraries

## 📧 Contact

For questions or collaboration, open an issue on GitHub.

---

**Status**: Active Development 🚧

Last updated: February 2026
