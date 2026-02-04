/*
 * MazeSolver Neural Network Controller v2.0 - ULTIMATE + FLOOD FILL
 * ===================================================================
 * Combines best features from all versions:
 * - Efficient NN inference with smart output mapping
 * - Encoder-based precise turns (not timed)
 * - FLOOD FILL algorithm for optimal path finding
 * - Maze mapping and distance calculation
 * - Dead-end detection and 180° recovery
 * - No emergency stops - continuous fast running
 * - Robust safety overrides without halting
 * 
 * Hardware:
 * - Teensy 4.1 @ 600MHz
 * - TB6612FNG Motor Driver
 * - 3x Sharp IR Sensors (Front, Left, Right)
 * - 2x Quadrature Encoders
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
#define TICKS_90 700          // Encoder ticks for 90 degree turn

// Flood fill maze parameters
#define MAZE_SIZE 16          // 16x16 maze (classic micromouse)
#define GOAL_X 7              // Goal center X (can be 7 or 8)
#define GOAL_Y 7              // Goal center Y (can be 7 or 8)
#define MAX_DISTANCE 255      // Maximum flood fill distance

// Safety thresholds (from calibration)
#define FRONT_EMERGENCY 2400  // Must turn
#define SIDE_CRASH 3600       // Touching wall
#define SIDE_DANGER 3200      // Very close to wall
#define SIDE_WALL 1800        // Wall detected
#define OPENING_THRESHOLD 1200 // Clear opening (no wall)
#define FRONT_CLEAR 1400      // Front is clear

// Speed limits
#define MIN_SPEED 60
#define MAX_SPEED 200
#define CRUISE_SPEED 160
#define TURN_SPEED 180
#define SLOW_SPEED 90

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
  TURNING_LEFT,
  TURNING_RIGHT,
  TURNING_180
};

State robotState = IDLE;
unsigned long lastUpdate = 0;
unsigned long startTime = 0;
int cellCount = 0;
bool sdReady = false;
int logNum = 0;

// Button debouncing
bool lastButtonState = HIGH;
bool buttonPressed = false;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

// Decision tracking for left-hand rule
bool preferLeft = true;

// ==================== MAZE & FLOOD FILL ====================
// Maze walls (bit flags: 0=North, 1=East, 2=South, 3=West)
uint8_t mazeWalls[MAZE_SIZE][MAZE_SIZE];

// Flood fill distances
uint8_t floodDist[MAZE_SIZE][MAZE_SIZE];

// Robot position and orientation
int8_t robotX = 0;         // Current X position
int8_t robotY = 0;         // Current Y position
int8_t robotDir = 0;       // 0=North, 1=East, 2=South, 3=West

// Goal tracking
bool goalReached = false;
bool returningToStart = false;
int8_t goalCenterX = GOAL_X;
int8_t goalCenterY = GOAL_Y;

// Decision tracking for left-hand rule
bool preferLeft = true;

// ==================== MAZE & FLOOD FILL ====================
// Maze walls (bit flags: 0=North, 1=East, 2=South, 3=West)
uint8_t mazeWalls[MAZE_SIZE][MAZE_SIZE];

// Flood fill distances
uint8_t floodDist[MAZE_SIZE][MAZE_SIZE];

// Robot position and orientation
int8_t robotX = 0;         // Current X position
int8_t robotY = 0;         // Current Y position
int8_t robotDir = 0;       // 0=North, 1=East, 2=South, 3=West

// Goal tracking
bool goalReached = false;
bool returningToStart = false;
int8_t goalCenterX = GOAL_X;
int8_t goalCenterY = GOAL_Y;

// Wall bit masks
#define WALL_NORTH 0x01
#define WALL_EAST  0x02
#define WALL_SOUTH 0x04
#define WALL_WEST  0x08

// Initialize maze (all walls unknown)
void initMaze() {
  for (int y = 0; y < MAZE_SIZE; y++) {
    for (int x = 0; x < MAZE_SIZE; x++) {
      mazeWalls[x][y] = 0;
      floodDist[x][y] = MAX_DISTANCE;
    }
  }
  
  // Set boundary walls
  for (int i = 0; i < MAZE_SIZE; i++) {
    mazeWalls[i][0] |= WALL_SOUTH;           // South boundary
    mazeWalls[i][MAZE_SIZE-1] |= WALL_NORTH; // North boundary
    mazeWalls[0][i] |= WALL_WEST;            // West boundary
    mazeWalls[MAZE_SIZE-1][i] |= WALL_EAST;  // East boundary
  }
  
  robotX = 0;
  robotY = 0;
  robotDir = 0;  // Facing North
  goalReached = false;
  returningToStart = false;
}

// Update walls based on sensor readings
void updateWalls() {
  // Front wall
  if (wallFront()) {
    switch (robotDir) {
      case 0: mazeWalls[robotX][robotY] |= WALL_NORTH; break;
      case 1: mazeWalls[robotX][robotY] |= WALL_EAST; break;
      case 2: mazeWalls[robotX][robotY] |= WALL_SOUTH; break;
      case 3: mazeWalls[robotX][robotY] |= WALL_WEST; break;
    }
  }
  
  // Left wall
  if (wallLeft()) {
    switch (robotDir) {
      case 0: mazeWalls[robotX][robotY] |= WALL_WEST; break;
      case 1: mazeWalls[robotX][robotY] |= WALL_NORTH; break;
      case 2: mazeWalls[robotX][robotY] |= WALL_EAST; break;
      case 3: mazeWalls[robotX][robotY] |= WALL_SOUTH; break;
    }
  }
  
  // Right wall
  if (wallRight()) {
    switch (robotDir) {
      case 0: mazeWalls[robotX][robotY] |= WALL_EAST; break;
      case 1: mazeWalls[robotX][robotY] |= WALL_SOUTH; break;
      case 2: mazeWalls[robotX][robotY] |= WALL_WEST; break;
      case 3: mazeWalls[robotX][robotY] |= WALL_NORTH; break;
    }
  }
}

// Flood fill algorithm - calculate distances from goal
void floodFill(int8_t targetX, int8_t targetY) {
  // Reset all distances
  for (int y = 0; y < MAZE_SIZE; y++) {
    for (int x = 0; x < MAZE_SIZE; x++) {
      floodDist[x][y] = MAX_DISTANCE;
    }
  }
  
  // Queue for BFS (simple array-based)
  int8_t queueX[MAZE_SIZE * MAZE_SIZE];
  int8_t queueY[MAZE_SIZE * MAZE_SIZE];
  int qHead = 0, qTail = 0;
  
  // Start from target
  floodDist[targetX][targetY] = 0;
  queueX[qTail] = targetX;
  queueY[qTail] = targetY;
  qTail++;
  
  // BFS to fill distances
  while (qHead < qTail) {
    int8_t cx = queueX[qHead];
    int8_t cy = queueY[qHead];
    qHead++;
    
    uint8_t currentDist = floodDist[cx][cy];
    
    // Check all 4 neighbors
    // North
    if (!(mazeWalls[cx][cy] & WALL_NORTH) && cy < MAZE_SIZE - 1) {
      if (floodDist[cx][cy+1] > currentDist + 1) {
        floodDist[cx][cy+1] = currentDist + 1;
        queueX[qTail] = cx;
        queueY[qTail] = cy + 1;
        qTail++;
      }
    }
    
    // East
    if (!(mazeWalls[cx][cy] & WALL_EAST) && cx < MAZE_SIZE - 1) {
      if (floodDist[cx+1][cy] > currentDist + 1) {
        floodDist[cx+1][cy] = currentDist + 1;
        queueX[qTail] = cx + 1;
        queueY[qTail] = cy;
        qTail++;
      }
    }
    
    // South
    if (!(mazeWalls[cx][cy] & WALL_SOUTH) && cy > 0) {
      if (floodDist[cx][cy-1] > currentDist + 1) {
        floodDist[cx][cy-1] = currentDist + 1;
        queueX[qTail] = cx;
        queueY[qTail] = cy - 1;
        qTail++;
      }
    }
    
    // West
    if (!(mazeWalls[cx][cy] & WALL_WEST) && cx > 0) {
      if (floodDist[cx-1][cy] > currentDist + 1) {
        floodDist[cx-1][cy] = currentDist + 1;
        queueX[qTail] = cx - 1;
        queueY[qTail] = cy;
        qTail++;
      }
    }
  }
}

// Get next best direction based on flood fill
int8_t getBestDirection() {
  uint8_t minDist = 255;
  int8_t bestDir = robotDir;
  
  // Check all 4 directions
  for (int8_t dir = 0; dir < 4; dir++) {
    // Check if wall in this direction
    bool hasWall = false;
    int8_t nx = robotX, ny = robotY;
    
    switch (dir) {
      case 0: // North
        hasWall = (mazeWalls[robotX][robotY] & WALL_NORTH);
        ny++;
        break;
      case 1: // East
        hasWall = (mazeWalls[robotX][robotY] & WALL_EAST);
        nx++;
        break;
      case 2: // South
        hasWall = (mazeWalls[robotX][robotY] & WALL_SOUTH);
        ny--;
        break;
      case 3: // West
        hasWall = (mazeWalls[robotX][robotY] & WALL_WEST);
        nx--;
        break;
    }
    
    // If no wall and lower distance, this is better
    if (!hasWall && nx >= 0 && nx < MAZE_SIZE && ny >= 0 && ny < MAZE_SIZE) {
      if (floodDist[nx][ny] < minDist) {
        minDist = floodDist[nx][ny];
        bestDir = dir;
      }
    }
  }
  
  return bestDir;
}

// Update robot position after moving forward
void updatePosition() {
  switch (robotDir) {
    case 0: robotY++; break; // North
    case 1: robotX++; break; // East
    case 2: robotY--; break; // South
    case 3: robotX--; break; // West
  }
  
  // Check if reached goal
  if (robotX == goalCenterX && robotY == goalCenterY && !goalReached) {
    goalReached = true;
    Serial.println("*** GOAL REACHED! ***");
  }
  
  // Check if back at start
  if (robotX == 0 && robotY == 0 && goalReached && returningToStart) {
    Serial.println("*** RETURNED TO START! ***");
    returningToStart = false;
  }
}

// Turn robot direction
void turnRobotDir(int8_t targetDir) {
  int8_t turnAmount = (targetDir - robotDir + 4) % 4;
  
  if (turnAmount == 1) {        // Turn right
    executeTurn(false);
    robotDir = (robotDir + 1) % 4;
  } else if (turnAmount == 3) { // Turn left
    executeTurn(true);
    robotDir = (robotDir + 3) % 4;
  } else if (turnAmount == 2) { // Turn 180
    executeTurn180();
    robotDir = (robotDir + 2) % 4;
  }
}

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
  
  // Map NN output [-1,1] to PWM using absolute value for magnitude
  // This ensures both wheels get positive forward speed
  motors.target_l = constrain((int)(fabs(nn_output[0]) * MAX_SPEED), CRUISE_SPEED, MAX_SPEED);
  motors.target_r = constrain((int)(fabs(nn_output[1]) * MAX_SPEED), CRUISE_SPEED, MAX_SPEED);
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

bool wallFront() { return sensors.front_filtered > SIDE_WALL; }
bool wallLeft() { return sensors.left_filtered > SIDE_WALL; }
bool wallRight() { return sensors.right_filtered > SIDE_WALL; }
bool openingLeft() { return sensors.left_filtered < OPENING_THRESHOLD; }
bool openingRight() { return sensors.right_filtered < OPENING_THRESHOLD; }
bool openingFront() { return sensors.front_filtered < FRONT_CLEAR; }

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
  delay(50);
}

// ==================== TURN FUNCTIONS ====================
void executeTurn(bool turnLeft) {
  logEvent(turnLeft ? "TURN_L" : "TURN_R");
  Serial.print("Executing turn: ");
  Serial.println(turnLeft ? "LEFT" : "RIGHT");
  
  stopMotors();
  delay(50);
  
  // Reset encoders
  encL.write(0);
  encR.write(0);
  
  // Set pivot turn direction
  if (turnLeft) {
    digitalWrite(AIN1, LOW); digitalWrite(AIN2, HIGH);  // Left backward
    digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW);  // Right forward
  } else {
    digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);  // Left forward
    digitalWrite(BIN1, LOW); digitalWrite(BIN2, HIGH);  // Right backward
  }
  
  // Ramp up to turn speed (faster)
  for (int spd = MIN_SPEED; spd <= TURN_SPEED; spd += 25) {
    analogWrite(PWMA, spd);
    analogWrite(PWMB, spd);
    delay(3);
  }
  
  // Execute turn
  uint32_t turnStart = millis();
  while (true) {
    long avgTicks = (abs(encL.read()) + abs(encR.read())) / 2;
    
    // Slow down near end
    if (avgTicks >= TICKS_90 - 150) {
      analogWrite(PWMA, SLOW_SPEED);
      analogWrite(PWMB, SLOW_SPEED);
    }
    
    if (avgTicks >= TICKS_90) break;
    if (millis() - turnStart > 800) break; // Timeout
    delay(3);
  }
  
  stopMotors();
  delay(30);
  
  // Reset to forward
  motorsForward();
  
  // Strong forward push to clear turn area
  analogWrite(PWMA, MAX_SPEED);
  analogWrite(PWMB, MAX_SPEED);
  encL.write(0);
  encR.write(0);
  while ((abs(encL.read()) + abs(encR.read())) / 2 < 150) delay(3);
  
  motors.pwm_l = CRUISE_SPEED;
  motors.pwm_r = CRUISE_SPEED;
}

void executeTurn180() {
  logEvent("TURN_180");
  Serial.println("Executing 180 turn (dead end)");
  
  stopMotors();
  delay(50);
  
  // Two 90 degree turns
  encL.write(0);
  encR.write(0);
  
  // Right turn (both motors same direction for pivot)
  digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, LOW); digitalWrite(BIN2, HIGH);
  
  // Ramp up (faster)
  for (int spd = MIN_SPEED; spd <= TURN_SPEED; spd += 25) {
    analogWrite(PWMA, spd);
    analogWrite(PWMB, spd);
    delay(3);
  }
  
  // Turn until 2x TICKS_90
  uint32_t turnStart = millis();
  while (true) {
    long avgTicks = (abs(encL.read()) + abs(encR.read())) / 2;
    if (avgTicks >= (TICKS_90 * 2) - 200) {
      analogWrite(PWMA, SLOW_SPEED);
      analogWrite(PWMB, SLOW_SPEED);
    }
    if (avgTicks >= TICKS_90 * 2) break;
    if (millis() - turnStart > 1500) break;
    delay(3);
  }
  
  stopMotors();
  delay(30);
  
  // Reset to forward
  motorsForward();
  analogWrite(PWMA, MAX_SPEED);
  analogWrite(PWMB, MAX_SPEED);
  encL.write(0);
  encR.write(0);
  while ((abs(encL.read()) + abs(encR.read())) / 2 < 150) delay(3);
  
  motors.pwm_l = CRUISE_SPEED;
  motors.pwm_r = CRUISE_SPEED;
}

// ==================== DECISION LOGIC (FLOOD FILL) ====================
void makeNavigationDecision() {
  // Update walls at current position
  updateWalls();
  
  // Determine target for flood fill
  int8_t targetX, targetY;
  if (!goalReached) {
    targetX = goalCenterX;
    targetY = goalCenterY;
  } else if (returningToStart) {
    targetX = 0;
    targetY = 0;
  } else {
    // At goal, decide to return or explore
    Serial.println("At goal - returning to start");
    returningToStart = true;
    targetX = 0;
    targetY = 0;
  }
  
  // Run flood fill from target
  floodFill(targetX, targetY);
  
  // Get best direction
  int8_t bestDir = getBestDirection();
  
  Serial.print("Pos: (");
  Serial.print(robotX);
  Serial.print(",");
  Serial.print(robotY);
  Serial.print(") Dir:");
  Serial.print(robotDir);
  Serial.print(" Best:");
  Serial.print(bestDir);
  Serial.print(" Dist:");
  Serial.println(floodDist[robotX][robotY]);
  
  // Turn to face best direction
  if (bestDir != robotDir) {
    turnRobotDir(bestDir);
  }
  
  // Continue running
  robotState = RUNNING;
}

// ==================== SAFETY OVERRIDES ====================
void applySafetyOverrides() {
  // NEVER STOP - only adjust steering
  
  // Critical: Crashing into left wall
  if (sensors.left_filtered > SIDE_CRASH) {
    motors.target_l = MAX_SPEED;
    motors.target_r = MIN_SPEED;
    Serial.println("SAFETY: Hard right!");
    return;
  }
  
  // Critical: Crashing into right wall  
  if (sensors.right_filtered > SIDE_CRASH) {
    motors.target_l = MIN_SPEED;
    motors.target_r = MAX_SPEED;
    Serial.println("SAFETY: Hard left!");
    return;
  }
  
  // Front wall critical - make decision
  if (sensors.front_filtered > FRONT_EMERGENCY) {
    Serial.println("SAFETY: Front critical - deciding...");
    stopMotors();
    delay(80);
    readSensors();
    makeNavigationDecision();
    return;
  }
  
  // Danger zone - gentle corrections
  if (sensors.left_filtered > SIDE_DANGER) {
    motors.target_l += 40;
    motors.target_r -= 20;
  }
  if (sensors.right_filtered > SIDE_DANGER) {
    motors.target_l -= 20;
    motors.target_r += 40;
  }
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
void logEvent(const char* event) {
  if (!sdReady || !dataFile) return;
  
  unsigned long elapsed = millis() - startTime;
  
  dataFile.print(elapsed);
  dataFile.print(",");
  dataFile.print(cellCount);
  dataFile.print(",");
  dataFile.print(sensors.front);
  dataFile.print(",");
  dataFile.print(sensors.left);
  dataFile.print(",");
  dataFile.print(sensors.right);
  dataFile.print(",");
  dataFile.print(motors.pwm_l);
  dataFile.print(",");
  dataFile.print(motors.pwm_r);
  dataFile.print(",");
  dataFile.println(event);
  dataFile.flush();
}

void logData() {
  if (!sdReady || !dataFile) return;
  
  unsigned long elapsed = millis() - startTime;
  
  dataFile.print(elapsed);
  dataFile.print(",");
  dataFile.print(cellCount);
  dataFile.print(",");
  dataFile.print(sensors.front);
  dataFile.print(",");
  dataFile.print(sensors.left);
  dataFile.print(",");
  dataFile.print(sensors.right);
  dataFile.print(",");
  dataFile.print(motors.pwm_l);
  dataFile.print(",");
  dataFile.print(motors.pwm_r);
  dataFile.print(",DRIVE");
  dataFile.println();
}

// ==================== DISPLAY ====================
void updateDisplay() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.setTextColor(SH110X_WHITE);
  
  // Title
  display.println("MazeSolver FF v2.0");
  display.println("------------------");
  
  // State
  display.print("State: ");
  switch (robotState) {
    case IDLE: display.println("IDLE"); break;
    case RUNNING: display.println("RUNNING"); break;
    case STOPPED: display.println("STOPPED"); break;
    case TURNING_LEFT: display.println("TURN_L"); break;
    case TURNING_RIGHT: display.println("TURN_R"); break;
    case TURNING_180: display.println("TURN_180"); break;
  }
  
  // Position and goal status
  display.print("Pos:(");
  display.print(robotX);
  display.print(",");
  display.print(robotY);
  display.print(") D:");
  display.println(robotDir);
  
  if (goalReached) {
    display.println("GOAL REACHED!");
  } else {
    display.print("Dist to goal: ");
    display.println(floodDist[robotX][robotY]);
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
  
  Serial.println("MazeSolver Neural Network + Flood Fill v2.0");
  Serial.println("============================================");
  
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
    
    // Find next log number
    while (logNum < 999) {
      char filename[32];
      sprintf(filename, "run_%03d.csv", logNum);
      if (!SD.exists(filename)) break;
      logNum++;
    }
    
    char filename[32];
    sprintf(filename, "run_%03d.csv", logNum);
    dataFile = SD.open(filename, FILE_WRITE);
    if (dataFile) {
      Serial.print("Logging to: ");
      Serial.println(filename);
      dataFile.println("ms,cells,f,l,r,pwmL,pwmR,event");
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
        
        // Initialize flood fill maze
        initMaze();
        
        logEvent("START");
        motorsForward();
        setMotors(CRUISE_SPEED, CRUISE_SPEED);
      }
      break;
      
    case RUNNING: {
      // Neural network inference for smooth control
      neuralNetworkInference();
      
      // Apply safety overrides (may trigger turns)
      applySafetyOverrides();
      
      // Apply motor commands
      setMotors(motors.target_l, motors.target_r);
      
      // Count cells and update position
      long avgEnc = (abs(encoders.left) + abs(encoders.right)) / 2;
      if (avgEnc > (cellCount + 1) * TICKS_PER_CELL) {
        cellCount++;
        updatePosition();  // Update maze position
        logEvent("CELL");
        Serial.print("Cell completed: ");
        Serial.println(cellCount);
      }
      
      // Log data every 100ms
      if (now - lastUpdate >= 100) {
        logData();
        lastUpdate = now;
      }
      
      // Stop button
      if (btnPress) {
        robotState = STOPPED;
        stopMotors();
        Serial.println("Stopped by user");
        logEvent("STOP");
      }
      break;
    }
      
    case STOPPED:
      stopMotors();
      if (btnPress) {
        robotState = IDLE;
        if (dataFile) {
          dataFile.close();
          logNum++;
        }
        Serial.println("Run ended. Press button to restart.");
      }
      break;
      
    default:
      break;
  }
  
  // Update display every 150ms
  static unsigned long lastDisplay = 0;
  if (now - lastDisplay >= 150) {
    updateDisplay();
    lastDisplay = now;
  }
  
  delay(3);  // Ultra-fast loop (~300Hz)
}
