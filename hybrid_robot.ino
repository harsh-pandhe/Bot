/*
 * Hybrid Robot Controller - Maze Solver + Line Follower
 * ======================================================
 * Mode 1: Neural Network Maze Solver (3 Sharp IR sensors)
 * Mode 2: 16-Channel Line Follower (multiplexed array)
 * 
 * Hardware:
 * - Teensy 4.1 @ 600MHz
 * - TB6612FNG Motor Driver
 * - 3x Sharp IR Sensors (Front, Left, Right) - Maze mode
 * - 16x IR Sensors via CD74HC4067 Multiplexer - Line mode
 * - 2x Quadrature Encoders
 * - SH1106 OLED Display
 * 
 * Controls:
 * - Button short press (IDLE): Cycle through modes
 * - Button long press (IDLE): Start selected mode
 * - Button press (RUNNING): Stop
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Encoder.h>
#include <SD.h>

// Include trained weights for maze solver
#include "weights.h"

// ==================== PIN DEFINITIONS ====================
// Motor driver pins (TB6612FNG) - SHARED
#define PWMA 9
#define PWMB 10
#define AIN1 4
#define AIN2 3
#define BIN1 6
#define BIN2 7
#define STBY 5

// Maze mode: Sharp IR sensors
const uint8_t SENSOR_F = 21;
const uint8_t SENSOR_L = 23;
const uint8_t SENSOR_R = 22;

// Line mode: Multiplexer pins
#define S0 14
#define S1 15
#define S2 16
#define S3 17
#define SIGNAL_PIN 20   // A6

// Encoder pins - SHARED
#define ENC_LA 1
#define ENC_LB 0
#define ENC_RA 2
#define ENC_RB 8

// Button and SD
#define BTN_PIN 11
#define SD_CS BUILTIN_SDCARD

// Display
#define DISPLAY_ADDR 0x3C
#define DISPLAY_WIDTH 128
#define DISPLAY_HEIGHT 64

// ==================== MODE SELECTION ====================
enum OperatingMode {
  MODE_LINE_FOLLOW,
  MODE_MAZE_SOLVE,
  MODE_AUTO_HYBRID  // Auto-switch: Line follow -> Maze when walls detected
};

enum State {
  IDLE,
  RUNNING,
  STOPPED
};

enum AutoHybridState {
  AUTO_LINE_MODE,
  AUTO_MAZE_MODE
};

OperatingMode currentMode = MODE_LINE_FOLLOW;
State robotState = IDLE;
AutoHybridState autoState = AUTO_LINE_MODE;

// ==================== SHARED OBJECTS ====================
Adafruit_SH1106G display(DISPLAY_WIDTH, DISPLAY_HEIGHT, &Wire, -1);
Encoder encL(ENC_LA, ENC_LB);
Encoder encR(ENC_RA, ENC_RB);
File dataFile;

// ==================== MAZE MODE CONSTANTS ====================
#define CELL_SIZE 180
#define TICKS_PER_CELL_MAZE 950
#define TICKS_90 700
#define FRONT_EMERGENCY 2400
#define SIDE_CRASH 3600
#define SIDE_DANGER 3200
#define SIDE_WALL 1800
#define OPENING_THRESHOLD 1200
#define FRONT_CLEAR 1400
#define MAZE_MIN_SPEED 80
#define MAZE_MAX_SPEED 255
#define MAZE_CRUISE_SPEED 200
#define MAZE_TURN_SPEED 220
#define MAZE_SLOW_SPEED 120
#define MEDIAN_SAMPLES 3
#define FILTER_ALPHA 0.3
#define STUCK_TIMEOUT 800        // ms without movement = stuck
#define STUCK_THRESHOLD 15       // minimum encoder ticks to consider moving
#define RECOVERY_REVERSE_TIME 300 // ms to reverse
#define RECOVERY_TURN_TIME 500    // ms to turn

// ==================== LINE MODE CONSTANTS ====================
#define NUM_SENSORS 16
#define LOOP_THRESHOLD 12
#define SHARP_CURVE_THRESHOLD 4
#define TURN_THRESHOLD 8
#define LOOP_SPEED 100
#define CURVE_SPEED 140
#define LINE_BASE_SPEED 220
#define START_SPEED 120
#define CAL_SPEED 180
#define ACCEL_STEP 4
#define TICKS_PER_CM 19.5
#define GRID_CELL_CM 30
#define TICKS_PER_CELL_LINE (GRID_CELL_CM * TICKS_PER_CM)

// ==================== MAZE MODE VARIABLES ====================
struct MazeSensors {
  int front, left, right;
  float front_filtered, left_filtered, right_filtered;
} mazeSensors;

// Neural network arrays
float nn_input[3];
float nn_h1[32];
float nn_h2[16];
float nn_output[2];

// Stuck detection
unsigned long lastMovementTime = 0;
long lastStuckCheckEncL = 0;
long lastStuckCheckEncR = 0;
bool inRecovery = false;

// ==================== LINE MODE VARIABLES ====================
bool isBlackLine = true;
bool lineCalibrated = false;
float Kp = 0.06;
float Kd = 1.0;
float Ki = 0.0;
int lineBaseSpeed = LINE_BASE_SPEED;
int currentSpeed = START_SPEED;
int targetSpeed = LINE_BASE_SPEED;
int activeSensors = 0;
int leftActiveSensors = 0;
int rightActiveSensors = 0;
bool loopDetected = false;
bool sharpCurveDetected = false;
bool atIntersection = false;

int sensorWeight[NUM_SENSORS] = {
  -7,-6,-5,-4,-3,-2,-1, 0,
   0, 1, 2, 3, 4, 5, 6, 7
};

int minVal[NUM_SENSORS], maxVal[NUM_SENSORS], threshold[NUM_SENSORS];
int sensorVal[NUM_SENSORS], sensorBin[NUM_SENSORS];
int P, D, I, prevError;
int lsp, rsp;
bool onLine;

// ==================== SHARED VARIABLES ====================
struct MotorControl {
  int pwm_l, pwm_r;
  int target_l, target_r;
} motors;

struct EncoderData {
  long left, right;
  long left_prev, right_prev;
} encoders;

long encLeft = 0;
long encRight = 0;
long lastCellDistance = 0;
int cellCount = 0;

unsigned long lastUpdate = 0;
unsigned long startTime = 0;
bool sdReady = false;
int logNum = 0;

// Button handling
bool lastButtonState = HIGH;
unsigned long buttonPressStart = 0;
bool buttonPressed = false;
#define LONG_PRESS_TIME 800

// ==================== NEURAL NETWORK FUNCTIONS ====================
float relu(float x) {
  return x > 0 ? x : 0;
}

float tanh_approx(float x) {
  if (x > 3) return 1.0;
  if (x < -3) return -1.0;
  float x2 = x * x;
  return x * (27 + x2) / (27 + 9 * x2);
}

void neuralNetworkInference() {
  nn_input[0] = mazeSensors.front_filtered / 4095.0;
  nn_input[1] = mazeSensors.left_filtered / 4095.0;
  nn_input[2] = mazeSensors.right_filtered / 4095.0;
  
  for (int i = 0; i < 32; i++) {
    float sum = b1[i];
    for (int j = 0; j < 3; j++) {
      sum += nn_input[j] * w1[i][j];
    }
    nn_h1[i] = tanh_approx(sum);
  }
  
  for (int i = 0; i < 16; i++) {
    float sum = b2[i];
    for (int j = 0; j < 32; j++) {
      sum += nn_h1[j] * w2[i][j];
    }
    nn_h2[i] = tanh_approx(sum);
  }
  
  for (int i = 0; i < 2; i++) {
    float sum = b3[i];
    for (int j = 0; j < 16; j++) {
      sum += nn_h2[j] * w3[i][j];
    }
    nn_output[i] = tanh_approx(sum);
  }
  
  motors.target_l = constrain((int)(fabs(nn_output[0]) * MAZE_MAX_SPEED), MAZE_CRUISE_SPEED, MAZE_MAX_SPEED);
  motors.target_r = constrain((int)(fabs(nn_output[1]) * MAZE_MAX_SPEED), MAZE_CRUISE_SPEED, MAZE_MAX_SPEED);
}

// ==================== MAZE SENSOR FUNCTIONS ====================
int readSensorMedian(int pin) {
  int samples[MEDIAN_SAMPLES];
  for (int i = 0; i < MEDIAN_SAMPLES; i++) {
    samples[i] = analogRead(pin);
    delayMicroseconds(100);
  }
  
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

void readMazeSensors() {
  mazeSensors.front = readSensorMedian(SENSOR_F);
  mazeSensors.left = readSensorMedian(SENSOR_L);
  mazeSensors.right = readSensorMedian(SENSOR_R);
  
  mazeSensors.front_filtered = FILTER_ALPHA * mazeSensors.front + (1 - FILTER_ALPHA) * mazeSensors.front_filtered;
  mazeSensors.left_filtered = FILTER_ALPHA * mazeSensors.left + (1 - FILTER_ALPHA) * mazeSensors.left_filtered;
  mazeSensors.right_filtered = FILTER_ALPHA * mazeSensors.right + (1 - FILTER_ALPHA) * mazeSensors.right_filtered;
}

bool wallFront() { return mazeSensors.front_filtered > SIDE_WALL; }
bool wallLeft() { return mazeSensors.left_filtered > SIDE_WALL; }
bool wallRight() { return mazeSensors.right_filtered > SIDE_WALL; }
bool openingLeft() { return mazeSensors.left_filtered < OPENING_THRESHOLD; }
bool openingRight() { return mazeSensors.right_filtered < OPENING_THRESHOLD; }
bool openingFront() { return mazeSensors.front_filtered < FRONT_CLEAR; }

// Check if robot has entered maze area (walls detected on sides)
bool detectMazeEntry() {
  // Detect maze when we have walls on both sides or multiple walls detected
  int wallCount = 0;
  if (wallLeft()) wallCount++;
  if (wallRight()) wallCount++;
  if (wallFront()) wallCount++;
  
  // If we detect 2+ walls, we're in a maze
  return wallCount >= 2;
}

// ==================== LINE SENSOR FUNCTIONS ====================
int sensorRead(int ch) {
  digitalWrite(S0, ch & 0x01);
  digitalWrite(S1, ch & 0x02);
  digitalWrite(S2, ch & 0x04);
  digitalWrite(S3, ch & 0x08);
  delayMicroseconds(3);
  return analogRead(SIGNAL_PIN);
}

void readLine() {
  onLine = false;
  activeSensors = 0;
  leftActiveSensors = 0;
  rightActiveSensors = 0;

  for (int i = 0; i < NUM_SENSORS; i++) {
    if (isBlackLine)
      sensorVal[i] = map(sensorRead(i), minVal[i], maxVal[i], 0, 1000);
    else
      sensorVal[i] = map(sensorRead(i), minVal[i], maxVal[i], 1000, 0);

    sensorVal[i] = constrain(sensorVal[i], 0, 1000);
    sensorBin[i] = sensorVal[i] > 500;

    if (sensorBin[i]) {
      onLine = true;
      activeSensors++;
      if (i < 8) rightActiveSensors++;
      else leftActiveSensors++;
    }
  }
}

void detectIntersections() {
  if (activeSensors >= TURN_THRESHOLD) {
    if (!atIntersection) {
      atIntersection = true;
      targetSpeed = LOOP_SPEED;
    }
  } else {
    atIntersection = false;
  }
}

void detectLoopAndCurves() {
  loopDetected = false;
  sharpCurveDetected = false;
  
  if (activeSensors >= LOOP_THRESHOLD) {
    loopDetected = true;
  }
  else if (rightActiveSensors >= SHARP_CURVE_THRESHOLD || 
           leftActiveSensors >= SHARP_CURVE_THRESHOLD) {
    sharpCurveDetected = true;
  }
}

void adaptiveSpeedControl() {
  if (loopDetected) {
    targetSpeed = LOOP_SPEED;
  } else if (sharpCurveDetected) {
    targetSpeed = CURVE_SPEED;
  } else if (activeSensors <= 2) {
    targetSpeed = lineBaseSpeed;
  } else {
    targetSpeed = lineBaseSpeed - 40;
  }
  
  if (currentSpeed < targetSpeed) {
    currentSpeed = min(targetSpeed, currentSpeed + ACCEL_STEP);
  } else if (currentSpeed > targetSpeed) {
    currentSpeed = max(targetSpeed, currentSpeed - (ACCEL_STEP * 2));
  }
}

// ==================== STUCK DETECTION & RECOVERY ====================
bool checkIfStuck() {
  unsigned long now = millis();
  
  // Get current encoder positions
  long currentEncL = encL.read();
  long currentEncR = encR.read();
  
  // Calculate movement since last check
  long deltaL = abs(currentEncL - lastStuckCheckEncL);
  long deltaR = abs(currentEncR - lastStuckCheckEncR);
  long totalMovement = deltaL + deltaR;
  
  // If robot is moving, update timer
  if (totalMovement > STUCK_THRESHOLD) {
    lastMovementTime = now;
    lastStuckCheckEncL = currentEncL;
    lastStuckCheckEncR = currentEncR;
    return false;
  }
  
  // Check if stuck timeout reached
  if (now - lastMovementTime > STUCK_TIMEOUT) {
    return true;
  }
  
  return false;
}

void executeRecovery() {
  Serial.println("*** STUCK DETECTED - EXECUTING RECOVERY ***");
  inRecovery = true;
  
  // Step 1: Reverse
  motorsReverse();
  setMotors(MAZE_SLOW_SPEED, MAZE_SLOW_SPEED);
  delay(RECOVERY_REVERSE_TIME);
  
  // Step 2: Turn (alternate direction based on sensor readings)
  if (mazeSensors.left_filtered > mazeSensors.right_filtered) {
    // More space on left, turn right
    motorsForward();
    setMotors(MAZE_TURN_SPEED, -MAZE_SLOW_SPEED);
  } else {
    // More space on right, turn left
    motorsForward();
    setMotors(-MAZE_SLOW_SPEED, MAZE_TURN_SPEED);
  }
  delay(RECOVERY_TURN_TIME);
  
  // Reset and continue
  stopMotors();
  delay(100);
  motorsForward();
  
  // Reset stuck detection
  lastMovementTime = millis();
  lastStuckCheckEncL = encL.read();
  lastStuckCheckEncR = encR.read();
  inRecovery = false;
  
  Serial.println("*** RECOVERY COMPLETE ***");
}

// ==================== MOTOR CONTROL ====================
void motorsForward() {
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, HIGH);
  digitalWrite(BIN2, LOW);
}

void setMotors(int left, int right) {
  left = constrain(left, 0, 255);
  right = constrain(right, 0, 255);
  
  if (left > 0 && left < MAZE_MIN_SPEED) left = MAZE_MIN_SPEED;
  if (right > 0 && right < MAZE_MIN_SPEED) right = MAZE_MIN_SPEED;
  
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
  delay(50);
}

void motorLeft(int spd) {
  spd = constrain(spd, -255, 255);
  if (spd > 0) {
    digitalWrite(AIN1, HIGH);
    digitalWrite(AIN2, LOW);
    analogWrite(PWMA, spd);
  } else if (spd < 0) {
    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, HIGH);
    analogWrite(PWMA, -spd);
  } else {
    digitalWrite(AIN1, HIGH);
    digitalWrite(AIN2, HIGH);
    analogWrite(PWMA, 0);
  }
}

void motorRight(int spd) {
  spd = constrain(spd, -255, 255);
  if (spd > 0) {
    digitalWrite(BIN1, HIGH);
    digitalWrite(BIN2, LOW);
    analogWrite(PWMB, spd);
  } else if (spd < 0) {
    digitalWrite(BIN1, LOW);
    digitalWrite(BIN2, HIGH);
    analogWrite(PWMB, -spd);
  } else {
    digitalWrite(BIN1, HIGH);
    digitalWrite(BIN2, HIGH);
    analogWrite(PWMB, 0);
  }
}

// ==================== MAZE TURN FUNCTIONS ====================
void executeTurn(bool turnLeft) {
  Serial.print("Executing turn: ");
  Serial.println(turnLeft ? "LEFT" : "RIGHT");
  
  stopMotors();
  delay(50);
  
  encL.write(0);
  encR.write(0);
  
  if (turnLeft) {
    digitalWrite(AIN1, LOW); digitalWrite(AIN2, HIGH);
    digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW);
  } else {
    digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
    digitalWrite(BIN1, LOW); digitalWrite(BIN2, HIGH);
  }
  
  for (int spd = MAZE_MIN_SPEED; spd <= MAZE_TURN_SPEED; spd += 25) {
    analogWrite(PWMA, spd);
    analogWrite(PWMB, spd);
    delay(3);
  }
  
  uint32_t turnStart = millis();
  while (true) {
    long avgTicks = (abs(encL.read()) + abs(encR.read())) / 2;
    
    if (avgTicks >= TICKS_90 - 150) {
      analogWrite(PWMA, MAZE_SLOW_SPEED);
      analogWrite(PWMB, MAZE_SLOW_SPEED);
    }
    
    if (avgTicks >= TICKS_90) break;
    if (millis() - turnStart > 800) break;
    delay(3);
  }
  
  stopMotors();
  delay(30);
  
  motorsForward();
  
  analogWrite(PWMA, MAZE_MAX_SPEED);
  analogWrite(PWMB, MAZE_MAX_SPEED);
  encL.write(0);
  encR.write(0);
  while ((abs(encL.read()) + abs(encR.read())) / 2 < 150) delay(3);
  
  motors.pwm_l = MAZE_CRUISE_SPEED;
  motors.pwm_r = MAZE_CRUISE_SPEED;
}

void executeTurn180() {
  Serial.println("Executing 180 turn (dead end)");
  
  stopMotors();
  delay(50);
  
  encL.write(0);
  encR.write(0);
  
  digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, LOW); digitalWrite(BIN2, HIGH);
  
  for (int spd = MAZE_MIN_SPEED; spd <= MAZE_TURN_SPEED; spd += 25) {
    analogWrite(PWMA, spd);
    analogWrite(PWMB, spd);
    delay(3);
  }
  
  uint32_t turnStart = millis();
  while (true) {
    long avgTicks = (abs(encL.read()) + abs(encR.read())) / 2;
    if (avgTicks >= (TICKS_90 * 2) - 200) {
      analogWrite(PWMA, MAZE_SLOW_SPEED);
      analogWrite(PWMB, MAZE_SLOW_SPEED);
    }
    if (avgTicks >= TICKS_90 * 2) break;
    if (millis() - turnStart > 1500) break;
    delay(3);
  }
  
  stopMotors();
  delay(30);
  
  motorsForward();
  analogWrite(PWMA, MAZE_MAX_SPEED);
  analogWrite(PWMB, MAZE_MAX_SPEED);
  encL.write(0);
  encR.write(0);
  while ((abs(encL.read()) + abs(encR.read())) / 2 < 150) delay(3);
  
  motors.pwm_l = MAZE_CRUISE_SPEED;
  motors.pwm_r = MAZE_CRUISE_SPEED;
}

// ==================== MAZE NAVIGATION ====================
void makeNavigationDecision() {
  bool hasFront = wallFront();
  bool hasLeft = wallLeft();
  bool hasRight = wallRight();
  bool leftOK = openingLeft();
  bool rightOK = openingRight();
  bool frontOK = openingFront();
  
  Serial.print("Walls: F=");
  Serial.print(hasFront ? "Y" : "N");
  Serial.print(" L=");
  Serial.print(hasLeft ? "Y" : "N");
  Serial.print(" R=");
  Serial.print(hasRight ? "Y" : "N");
  
  if (hasFront && hasLeft && hasRight) {
    Serial.println(" -> DEAD END (180)");
    executeTurn180();
    return;
  }
  
  if (frontOK && !hasFront) {
    Serial.println(" -> STRAIGHT (NN)");
    return;
  }
  
  if (hasFront && leftOK && rightOK) {
    Serial.println(" -> T-JUNCTION: LEFT");
    executeTurn(true);
    return;
  }
  
  if (leftOK) {
    Serial.println(" -> LEFT");
    executeTurn(true);
    return;
  }
  
  if (rightOK) {
    Serial.println(" -> RIGHT");
    executeTurn(false);
    return;
  }
  
  if (hasFront) {
    if (mazeSensors.left_filtered < mazeSensors.right_filtered) {
      Serial.println(" -> FORCED LEFT");
      executeTurn(true);
    } else {
      Serial.println(" -> FORCED RIGHT");
      executeTurn(false);
    }
    return;
  }
}

void applySafetyOverrides() {
  if (mazeSensors.left_filtered > SIDE_CRASH) {
    motors.target_l = MAZE_MAX_SPEED;
    motors.target_r = MAZE_MIN_SPEED;
    Serial.println("SAFETY: Hard right!");
    return;
  }
  
  if (mazeSensors.right_filtered > SIDE_CRASH) {
    motors.target_l = MAZE_MIN_SPEED;
    motors.target_r = MAZE_MAX_SPEED;
    Serial.println("SAFETY: Hard left!");
    return;
  }
  
  if (mazeSensors.front_filtered > FRONT_EMERGENCY) {
    Serial.println("SAFETY: Front critical - deciding...");
    stopMotors();
    delay(80);
    readMazeSensors();
    makeNavigationDecision();
    return;
  }
  
  if (mazeSensors.left_filtered > SIDE_DANGER) {
    motors.target_l += 40;
    motors.target_r -= 20;
  }
  if (mazeSensors.right_filtered > SIDE_DANGER) {
    motors.target_l -= 20;
    motors.target_r += 40;
  }
}

// ==================== LINE CALIBRATION ====================
void autoCalibrate() {
  Serial.println("=== LINE CALIBRATION START ===");

  for (int i = 0; i < NUM_SENSORS; i++) {
    minVal[i] = sensorRead(i);
    maxVal[i] = sensorRead(i);
  }

  for (int s = 0; s <= CAL_SPEED; s += 30) {
    motorLeft(s);
    motorRight(-s);
    delay(80);
  }

  startTime = millis();
  while (millis() - startTime < 5000) {
    motorLeft(CAL_SPEED);
    motorRight(-CAL_SPEED);

    for (int i = 0; i < NUM_SENSORS; i++) {
      int v = sensorRead(i);
      if (v < minVal[i]) minVal[i] = v;
      if (v > maxVal[i]) maxVal[i] = v;
    }
    
    if ((millis() - startTime) % 1000 < 50) {
      Serial.print(".");
    }
  }

  for (int s = CAL_SPEED; s >= 0; s -= 30) {
    motorLeft(s);
    motorRight(-s);
    delay(60);
  }

  motorLeft(0);
  motorRight(0);

  Serial.println("\nCalibration DONE");
  
  for (int i = 0; i < NUM_SENSORS; i++) {
    threshold[i] = (minVal[i] + maxVal[i]) / 2;
  }
  
  lineCalibrated = true;
}

// ==================== LINE FOLLOWING ====================
void lineFollowWithEncoders() {
  int error = 0;
  int active = 0;

  for (int i = 0; i < NUM_SENSORS; i++) {
    if (sensorBin[i]) {
      error += sensorWeight[i] * sensorVal[i];
      active++;
    }
  }

  if (active == 0) return;
  error /= active;

  P = error;
  I += error;
  D = error - prevError;
  prevError = error;

  int pid = (Kp * P) + (Ki * I) + (Kd * D);

  lsp = currentSpeed + pid;
  rsp = currentSpeed - pid;

  long encDiff = encLeft - encRight;
  int balanceCorrection = constrain(encDiff * 0.05, -15, 15);
  
  lsp -= balanceCorrection;
  rsp += balanceCorrection;

  lsp = constrain(lsp, -255, 255);
  rsp = constrain(rsp, -255, 255);

  motorLeft(lsp);
  motorRight(rsp);
}

// ==================== ENCODER FUNCTIONS ====================
void updateEncoders() {
  encLeft = encL.read();
  encRight = encR.read();
  
  long avgDistance = (abs(encLeft) + abs(encRight)) / 2;
  
  if (currentMode == MODE_LINE_FOLLOW) {
    if (avgDistance - lastCellDistance > TICKS_PER_CELL_LINE) {
      cellCount++;
      lastCellDistance = avgDistance;
    }
  } else {
    if (avgDistance > (cellCount + 1) * TICKS_PER_CELL_MAZE) {
      cellCount++;
      Serial.print("Cell completed: ");
      Serial.println(cellCount);
    }
  }
}

void resetEncoders() {
  encL.write(0);
  encR.write(0);
  encLeft = 0;
  encRight = 0;
  lastCellDistance = 0;
  cellCount = 0;
}

// ==================== DISPLAY ====================
void updateDisplay() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.setTextColor(SH110X_WHITE);
  
  // Title with mode
  display.println("HYBRID ROBOT v1.0");
  display.println("==================");
  
  // Current mode
  display.print("MODE: ");
  if (currentMode == MODE_LINE_FOLLOW) {
    display.println("LINE FOLLOW");
  } else if (currentMode == MODE_MAZE_SOLVE) {
    display.println("MAZE SOLVE");
  } else {
    display.println("AUTO HYBRID");
    if (robotState == RUNNING) {
      display.print("  Sub: ");
      display.println(autoState == AUTO_LINE_MODE ? "LINE" : "MAZE");
    }
  }
  
  // State
  display.print("State: ");
  switch (robotState) {
    case IDLE: 
      display.println("IDLE");
      display.println();
      display.println("Short: Change Mode");
      display.println("Long:  Start");
      break;
    case RUNNING: 
      display.println("RUNNING");
      
      if (currentMode == MODE_MAZE_SOLVE || (currentMode == MODE_AUTO_HYBRID && autoState == AUTO_MAZE_MODE)) {
        display.print("F:");
        display.print(mazeSensors.front);
        display.print(" L:");
        display.print(mazeSensors.left);
        display.print(" R:");
        display.println(mazeSensors.right);
      } else {
        display.print("Sens:");
        display.print(activeSensors);
        display.print(" L:");
        display.print(leftActiveSensors);
        display.print(" R:");
        display.println(rightActiveSensors);
        
        // Show wall detection status in AUTO mode
        if (currentMode == MODE_AUTO_HYBRID) {
          display.print("Walls: F");
          display.print(mazeSensors.front > SIDE_WALL ? "Y" : "N");
          display.print(" L");
          display.print(mazeSensors.left > SIDE_WALL ? "Y" : "N");
          display.print(" R");
          display.println(mazeSensors.right > SIDE_WALL ? "Y" : "N");
        }
      }
      
      display.print("PWM L:");
      display.print(motors.pwm_l);
      display.print(" R:");
      display.println(motors.pwm_r);
      
      display.print("Cells: ");
      display.println(cellCount);
      break;
    case STOPPED: 
      display.println("STOPPED");
      display.println("Press to reset");
      break;
  }
  
  display.display();
}

// ==================== BUTTON HANDLING ====================
void handleButton() {
  bool reading = digitalRead(BTN_PIN);
  
  // Detect button press
  if (lastButtonState == HIGH && reading == LOW) {
    buttonPressStart = millis();
    buttonPressed = true;
  }
  
  // Detect button release
  if (lastButtonState == LOW && reading == HIGH) {
    if (buttonPressed) {
      unsigned long pressDuration = millis() - buttonPressStart;
      
      if (robotState == IDLE) {
        if (pressDuration < LONG_PRESS_TIME) {
          // Short press: cycle mode
          if (currentMode == MODE_LINE_FOLLOW) {
            currentMode = MODE_MAZE_SOLVE;
            Serial.println("Mode: MAZE SOLVE");
          } else if (currentMode == MODE_MAZE_SOLVE) {
            currentMode = MODE_AUTO_HYBRID;
            Serial.println("Mode: AUTO HYBRID (Line->Maze)");
          } else {
            currentMode = MODE_LINE_FOLLOW;
            Serial.println("Mode: LINE FOLLOW");
          }
          updateDisplay();
        } else {
          // Long press: start
          Serial.print("Starting mode: ");
          if (currentMode == MODE_LINE_FOLLOW) {
            Serial.println("LINE FOLLOW");
          } else if (currentMode == MODE_MAZE_SOLVE) {
            Serial.println("MAZE SOLVE");
          } else {
            Serial.println("AUTO HYBRID (starts with LINE)");
            autoState = AUTO_LINE_MODE;
          }
          
          if ((currentMode == MODE_LINE_FOLLOW || currentMode == MODE_AUTO_HYBRID) && !lineCalibrated) {
            autoCalibrate();
          }
          
          robotState = RUNNING;
          startTime = millis();
          resetEncoders();
          cellCount = 0;
          
          if (currentMode == MODE_LINE_FOLLOW || currentMode == MODE_AUTO_HYBRID) {
            currentSpeed = START_SPEED;
            targetSpeed = lineBaseSpeed;
            I = 0;
            prevError = 0;
          }
          
          motorsForward();
          setMotors(MAZE_CRUISE_SPEED, MAZE_CRUISE_SPEED);
        }
      } else if (robotState == RUNNING) {
        // Stop
        Serial.println("Stopped by user");
        robotState = STOPPED;
        stopMotors();
      } else if (robotState == STOPPED) {
        // Reset to idle
        robotState = IDLE;
        lineCalibrated = false;
        Serial.println("Reset to IDLE");
      }
    }
    
    buttonPressed = false;
  }
  
  lastButtonState = reading;
}

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  delay(500);
  
  Serial.println("HYBRID ROBOT - Maze Solver + Line Follower");
  Serial.println("==========================================");
  
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
  
  // Multiplexer pins for line mode
  pinMode(S0, OUTPUT);
  pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);
  
  digitalWrite(STBY, HIGH);
  
  // Initialize ADC
  analogReadResolution(12);
  
  // Initialize sensor filters for maze mode
  for (int i = 0; i < 20; i++) {
    mazeSensors.front_filtered = analogRead(SENSOR_F);
    mazeSensors.left_filtered = analogRead(SENSOR_L);
    mazeSensors.right_filtered = analogRead(SENSOR_R);
    delay(5);
  }
  
  motorsForward();
  stopMotors();
  
  Serial.println("Ready!");
  Serial.println("Short press: Change mode");
  Serial.println("Long press: Start selected mode");
  
  updateDisplay();
}

// ==================== MAIN LOOP ====================
void loop() {
  unsigned long now = millis();
  
  // Handle button input
  handleButton();
  
  // Update display periodically
  static unsigned long lastDisplay = 0;
  if (now - lastDisplay >= 150) {
    updateDisplay();
    lastDisplay = now;
  }
  
  // State machine
  if (robotState != RUNNING) {
    delay(10);
    return;
  }
  
  // Update encoders
  updateEncoders();
  
  // Mode-specific execution
  if (currentMode == MODE_MAZE_SOLVE) {
    // ========== MAZE MODE ==========
    readMazeSensors();
    
    // Check if stuck
    if (!inRecovery && checkIfStuck()) {
      executeRecovery();
    }
    
    neuralNetworkInference();
    applySafetyOverrides();
    setMotors(motors.target_l, motors.target_r);
    
  } else if (currentMode == MODE_LINE_FOLLOW) {
    // ========== LINE FOLLOW MODE ==========
    readLine();
    detectIntersections();
    detectLoopAndCurves();
    adaptiveSpeedControl();
    
    if (onLine) {
      lineFollowWithEncoders();
    } else {
      if (prevError > 0) {
        motorLeft(-constrain(lineBaseSpeed * 0.55, 80, 255));
        motorRight(constrain(lineBaseSpeed, 80, 255));
      } else {
        motorLeft(constrain(lineBaseSpeed, 80, 255));
        motorRight(-constrain(lineBaseSpeed * 0.55, 80, 255));
      }
    }
    
  } else {
    // ========== AUTO HYBRID MODE ==========
    
    if (autoState == AUTO_LINE_MODE) {
      // Line following with wall monitoring
      readMazeSensors();  // Monitor walls
      readLine();
      detectIntersections();
      detectLoopAndCurves();
      adaptiveSpeedControl();
      
      // Check for maze entry
      if (detectMazeEntry()) {
        Serial.println("\n*** MAZE DETECTED - SWITCHING TO MAZE MODE ***\n");
        autoState = AUTO_MAZE_MODE;
        stopMotors();
        delay(200);
        motorsForward();
        setMotors(MAZE_CRUISE_SPEED, MAZE_CRUISE_SPEED);
      } else {
        // Continue line following
        if (onLine) {
          lineFollowWithEncoders();
        } else {
          if (prevError > 0) {
            motorLeft(-constrain(lineBaseSpeed * 0.55, 80, 255));
            motorRight(constrain(lineBaseSpeed, 80, 255));
          } else {
            motorLeft(constrain(lineBaseSpeed, 80, 255));
            motorRight(-constrain(lineBaseSpeed * 0.55, 80, 255));
          }
        }
      }
      
    } else {
      // AUTO_MAZE_MODE - Use maze solving
      readMazeSensors();
      
      // Check if stuck
      if (!inRecovery && checkIfStuck()) {
        executeRecovery();
      }
      
      neuralNetworkInference();
      applySafetyOverrides();
      setMotors(motors.target_l, motors.target_r);
    }
  }
  
  delay(3);
}
