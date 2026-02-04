/*
 * MazeSolver Neural Network Controller v1.0
 * ==========================================
 * Trained on comprehensive dataset from all collected runs
 * Uses 3-layer neural network: 3 -> 32 -> 16 -> 2
 * 
 * Hardware:
 * - Teensy 4.1
 * - TB6612FNG Motor Driver
 * - 3x Sharp IR Sensors (Front, Left, Right)
 * - 2x Quadrature Encoders
 * 
 * Features:
 * - Pure neural network control (no manual PID)
 * - Learned from successful maze runs
 * - Safety overrides for critical situations
 * - SD card data logging
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Encoder.h>
#include <SD.h>

// Include trained weights
#include "weights.h"

// ==================== PIN DEFINITIONS ====================
// Motor driver pins (TB6612FNG)
#define PWMA 9
#define PWMB 10
#define AIN1 4
#define AIN2 3
#define BIN1 6
#define BIN2 7
#define STBY 5

// Sensor pins (Sharp IR)
const uint8_t SENSOR_F = 21;
const uint8_t SENSOR_L = 23;
const uint8_t SENSOR_R = 22;

// Button and SD
#define BTN_PIN 11
#define SD_CS BUILTIN_SDCARD

// ==================== CONSTANTS ====================
#define DISPLAY_ADDR 0x3C
#define DISPLAY_WIDTH 128
#define DISPLAY_HEIGHT 64

// Maze parameters
#define CELL_SIZE 180         // mm
#define TICKS_PER_CELL 950    // Encoder ticks for one cell (calibrated)

// Safety thresholds (from calibration)
#define FRONT_EMERGENCY 2800  // Must stop immediately
#define SIDE_CRASH 3600       // Touching wall
#define SIDE_DANGER 3200      // Very close to wall
#define SIDE_WALL 1800        // Wall detected

// Speed limits
#define MIN_SPEED 50
#define MAX_SPEED 180
#define EMERGENCY_SPEED 30

// Filtering
#define MEDIAN_SAMPLES 3
#define FILTER_ALPHA 0.3

// ==================== GLOBAL OBJECTS ====================
Adafruit_SH1106G display(DISPLAY_WIDTH, DISPLAY_HEIGHT, &Wire, -1);
Encoder encL(1, 0);  // Left encoder pins
Encoder encR(2, 8);  // Right encoder pins
File dataFile;

// ==================== STATE VARIABLES ====================
struct SensorData {
  int front, left, right;
  float front_filtered, left_filtered, right_filtered;
} sensors;

struct MotorControl {
  int pwm_l, pwm_r;
  int target_l, target_r;
} motors;

struct EncoderData {
  long left, right;
  long left_prev, right_prev;
} encoders;

enum State {
  IDLE,
  RUNNING,
  STOPPED,
  EMERGENCY
};

State robotState = IDLE;
unsigned long lastUpdate = 0;
unsigned long startTime = 0;
int cellCount = 0;
bool sdReady = false;

// Button debouncing
bool lastButtonState = HIGH;
bool buttonPressed = false;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

// ==================== NEURAL NETWORK ====================
float nn_input[3];
float nn_h1[32];
float nn_h2[16];
float nn_output[2];

// Activation functions
float relu(float x) {
  return x > 0 ? x : 0;
}

float tanh_approx(float x) {
  // Fast tanh approximation
  if (x > 3) return 1.0;
  if (x < -3) return -1.0;
  float x2 = x * x;
  return x * (27 + x2) / (27 + 9 * x2);
}

// Neural network forward pass
void neuralNetworkInference() {
  // Normalize inputs [0, 4095] -> [0, 1]
  nn_input[0] = sensors.front_filtered / SENSOR_MAX;
  nn_input[1] = sensors.left_filtered / SENSOR_MAX;
  nn_input[2] = sensors.right_filtered / SENSOR_MAX;
  
  // Layer 1: Input -> Hidden1 (3 -> 32)
  for (int i = 0; i < 32; i++) {
    float sum = b1[i];
    for (int j = 0; j < 3; j++) {
      sum += nn_input[j] * w1[i][j];
    }
    nn_h1[i] = tanh_approx(sum);
  }
  
  // Layer 2: Hidden1 -> Hidden2 (32 -> 16)
  for (int i = 0; i < 16; i++) {
    float sum = b2[i];
    for (int j = 0; j < 32; j++) {
      sum += nn_h1[j] * w2[i][j];
    }
    nn_h2[i] = tanh_approx(sum);
  }
  
  // Layer 3: Hidden2 -> Output (16 -> 2)
  for (int i = 0; i < 2; i++) {
    float sum = b3[i];
    for (int j = 0; j < 16; j++) {
      sum += nn_h2[j] * w3[i][j];
    }
    nn_output[i] = tanh_approx(sum);  // Output in [-1, 1]
  }
  
  // Denormalize outputs to PWM range [0, 255]
  motors.target_l = constrain((int)(nn_output[0] * PWM_MAX), 0, MAX_SPEED);
  motors.target_r = constrain((int)(nn_output[1] * PWM_MAX), 0, MAX_SPEED);
}

// ==================== SENSOR READING ====================
int readSensorMedian(int pin) {
  int samples[MEDIAN_SAMPLES];
  for (int i = 0; i < MEDIAN_SAMPLES; i++) {
    samples[i] = analogRead(pin);
    delayMicroseconds(100);
  }
  
  // Bubble sort
  for (int i = 0; i < MEDIAN_SAMPLES - 1; i++) {
    for (int j = 0; j < MEDIAN_SAMPLES - i - 1; j++) {
      if (samples[j] > samples[j + 1]) {
        int temp = samples[j];
        samples[j] = samples[j + 1];
        samples[j + 1] = temp;
      }
    }
  }
  
  return samples[MEDIAN_SAMPLES / 2];
}

void readSensors() {
  // Read raw values with median filter
  sensors.front = readSensorMedian(SENSOR_F);
  sensors.left = readSensorMedian(SENSOR_L);
  sensors.right = readSensorMedian(SENSOR_R);
  
  // Apply exponential moving average filter
  sensors.front_filtered = FILTER_ALPHA * sensors.front + (1 - FILTER_ALPHA) * sensors.front_filtered;
  sensors.left_filtered = FILTER_ALPHA * sensors.left + (1 - FILTER_ALPHA) * sensors.left_filtered;
  sensors.right_filtered = FILTER_ALPHA * sensors.right + (1 - FILTER_ALPHA) * sensors.right_filtered;
}

// ==================== MOTOR CONTROL ====================
void motorsForward() {
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, HIGH);
  digitalWrite(BIN2, LOW);
}

void setMotors(int left, int right) {
  // Constrain to safe range
  left = constrain(left, 0, MAX_SPEED);
  right = constrain(right, 0, MAX_SPEED);
  
  // Apply minimum speed threshold
  if (left > 0 && left < MIN_SPEED) left = MIN_SPEED;
  if (right > 0 && right < MIN_SPEED) right = MIN_SPEED;
  
  analogWrite(PWMA, left);
  analogWrite(PWMB, right);
  
  motors.pwm_l = left;
  motors.pwm_r = right;
}

void stopMotors() {
  analogWrite(PWMA, 0);
  analogWrite(PWMB, 0);
  motors.pwm_l = 0;
  motors.pwm_r = 0;
}

// ==================== SAFETY OVERRIDES ====================
bool checkSafety() {
  // Emergency: Front wall too close
  if (sensors.front_filtered > FRONT_EMERGENCY) {
    setMotors(EMERGENCY_SPEED, EMERGENCY_SPEED);
    robotState = EMERGENCY;
    return false;
  }
  
  // Critical: Touching left wall
  if (sensors.left_filtered > SIDE_CRASH) {
    // Override NN: steer hard right
    int emergency_l = motors.target_l + 80;
    int emergency_r = motors.target_r - 80;
    setMotors(emergency_l, emergency_r);
    return false;
  }
  
  // Critical: Touching right wall
  if (sensors.right_filtered > SIDE_CRASH) {
    // Override NN: steer hard left
    int emergency_l = motors.target_l - 80;
    int emergency_r = motors.target_r + 80;
    setMotors(emergency_l, emergency_r);
    return false;
  }
  
  return true;  // All clear
}

// ==================== ENCODER HANDLING ====================
void updateEncoders() {
  encoders.left = encL.read();
  encoders.right = encR.read();
}

void resetEncoders() {
  encL.write(0);
  encR.write(0);
  encoders.left = 0;
  encoders.right = 0;
  encoders.left_prev = 0;
  encoders.right_prev = 0;
}

// ==================== DATA LOGGING ====================
void logData() {
  if (!sdReady || !dataFile) return;
  
  unsigned long elapsed = millis() - startTime;
  
  dataFile.print(elapsed);
  dataFile.print(",");
  dataFile.print(sensors.front);
  dataFile.print(",");
  dataFile.print(sensors.left);
  dataFile.print(",");
  dataFile.print(sensors.right);
  dataFile.print(",");
  dataFile.print(encoders.left);
  dataFile.print(",");
  dataFile.print(encoders.right);
  dataFile.print(",");
  dataFile.print(motors.pwm_l);
  dataFile.print(",");
  dataFile.print(motors.pwm_r);
  dataFile.print(",");
  dataFile.println(robotState);
  
  dataFile.flush();
}

// ==================== DISPLAY ====================
void updateDisplay() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.setTextColor(SH110X_WHITE);
  
  // Title
  display.println("MazeSolver NN v1.0");
  display.println("------------------");
  
  // State
  display.print("State: ");
  switch (robotState) {
    case IDLE: display.println("IDLE"); break;
    case RUNNING: display.println("RUNNING"); break;
    case STOPPED: display.println("STOPPED"); break;
    case EMERGENCY: display.println("EMERGENCY"); break;
  }
  
  // Sensors
  display.print("F:");
  display.print(sensors.front);
  display.print(" L:");
  display.print(sensors.left);
  display.print(" R:");
  display.println(sensors.right);
  
  // Motors
  display.print("PWM L:");
  display.print(motors.pwm_l);
  display.print(" R:");
  display.println(motors.pwm_r);
  
  // Encoders
  display.print("Enc L:");
  display.print(encoders.left);
  display.print(" R:");
  display.println(encoders.right);
  
  // Cells
  display.print("Cells: ");
  display.println(cellCount);
  
  display.display();
}

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  delay(500);
  
  Serial.println("MazeSolver Neural Network v1.0");
  Serial.println("===============================");
  
  // Initialize display
  if (!display.begin(DISPLAY_ADDR, true)) {
    Serial.println("Display init failed!");
  }
  display.clearDisplay();
  display.display();
  
  // Initialize pins
  pinMode(STBY, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);
  pinMode(PWMA, OUTPUT);
  pinMode(PWMB, OUTPUT);
  pinMode(BTN_PIN, INPUT_PULLUP);
  
  // Enable motor driver
  digitalWrite(STBY, HIGH);
  
  // Initialize sensors
  analogReadResolution(12);  // 12-bit ADC
  
  // Initialize SD card
  if (SD.begin(SD_CS)) {
    Serial.println("SD card ready");
    sdReady = true;
    
    // Create new log file
    int fileNum = 0;
    char filename[32];
    do {
      sprintf(filename, "run_%03d.csv", fileNum++);
    } while (SD.exists(filename));
    
    dataFile = SD.open(filename, FILE_WRITE);
    if (dataFile) {
      Serial.print("Logging to: ");
      Serial.println(filename);
      dataFile.println("time_ms,front,left,right,enc_l,enc_r,pwm_l,pwm_r,state");
    }
  } else {
    Serial.println("SD card failed");
  }
  
  // Initialize sensor filters (prime with multiple readings)
  for (int i = 0; i < 20; i++) {
    sensors.front_filtered = analogRead(SENSOR_F);
    sensors.left_filtered = analogRead(SENSOR_L);
    sensors.right_filtered = analogRead(SENSOR_R);
    delay(5);
  }
  
  // Set motor direction to forward
  motorsForward();
  
  stopMotors();
  
  Serial.println("Ready! Press button to start.");
  updateDisplay();
}

// ==================== BUTTON HANDLING ====================
bool checkButtonPress() {
  bool reading = digitalRead(BTN_PIN);
  
  // If button state changed, reset debounce timer
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }
  
  bool pressed = false;
  
  // If stable for debounceDelay, register press
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading == LOW && !buttonPressed) {
      pressed = true;
      buttonPressed = true;
    } else if (reading == HIGH) {
      buttonPressed = false;
    }
  }
  
  lastButtonState = reading;
  return pressed;
}

// ==================== MAIN LOOP ====================
void loop() {
  unsigned long now = millis();
  
  // Read sensors every cycle
  readSensors();
  updateEncoders();
  
  // Check button
  bool btnPress = checkButtonPress();
  
  // State machine
  switch (robotState) {
    case IDLE:
      if (btnPress) {
        Serial.println("Starting maze run...");
        robotState = RUNNING;
        startTime = millis();
        resetEncoders();
        cellCount = 0;
      }
      break;
      
    case RUNNING: {
      // Neural network inference
      neuralNetworkInference();
      
      // Safety checks (may override NN output)
      bool safe = checkSafety();
      
      // Apply motor commands (if safe, use NN output)
      if (safe) {
        setMotors(motors.target_l, motors.target_r);
      }
      
      // Count cells
      long avgEnc = (encoders.left + encoders.right) / 2;
      if (avgEnc > (cellCount + 1) * TICKS_PER_CELL) {
        cellCount++;
        Serial.print("Cell completed: ");
        Serial.println(cellCount);
      }
      
      // Log data every 50ms
      if (now - lastUpdate >= 50) {
        logData();
        lastUpdate = now;
      }
      
      // Stop button
      if (btnPress) {
        robotState = STOPPED;
        stopMotors();
        Serial.println("Stopped by user");
      }
      
      // Emergency recovery
      if (robotState == EMERGENCY) {
        delay(500);  // Brief pause
        if (sensors.front_filtered < FRONT_EMERGENCY - 300) {
          robotState = RUNNING;  // Resume
        }
      }
      break;
    }
      
    case STOPPED:
    case EMERGENCY:
      stopMotors();
      if (btnPress) {
        robotState = IDLE;
        if (dataFile) {
          dataFile.close();
        }
        Serial.println("Run ended. Press button to restart.");
      }
      break;
  }
  
  // Update display every 100ms
  static unsigned long lastDisplay = 0;
  if (now - lastDisplay >= 100) {
    updateDisplay();
    lastDisplay = now;
  }
  
  delay(10);  // 100Hz main loop
}
