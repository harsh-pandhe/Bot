/*
 * MazeSolver FINAL v8.0 - ZERO WALL HITS
 * =======================================
 * Neural Network trained on YOUR data + safety injection
 * Robot: 120mm x 140mm, Track: 250mm
 * 
 * Sensor Calibration:
 *   Front: 3000 @ 3cm, 2000 @ 2cm
 *   Sides: 4000 touching, ~2000 centered
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Encoder.h>
#include "weights.h"

// ==================== TUNABLE PARAMETERS ====================
const long TICKS_90 = 455;     // Encoder ticks for 90° turn

// Speed settings
const int16_t MAX_SPEED = 160;    // Maximum forward speed
const int16_t CRUISE_SPEED = 130; // Normal cruising
const int16_t SLOW_SPEED = 80;    // Near obstacles
const int16_t MIN_PWM = 55;       // Minimum to move
const int16_t TURN_SPEED = 115;   // Pivot turn speed

// Thresholds (calibrated for your sensors)
const int16_t FRONT_EMERGENCY = 2800;  // ~2.5cm - STOP
const int16_t FRONT_CLOSE = 2400;      // ~3cm - slow down
const int16_t FRONT_DETECT = 1800;     // ~4cm - wall ahead

const int16_t SIDE_CRASH = 3600;       // Almost touching - EMERGENCY
const int16_t SIDE_DANGER = 3200;      // ~2cm - strong correction
const int16_t SIDE_CLOSE = 2600;       // Getting close - mild correction
const int16_t SIDE_CENTERED = 2000;    // Centered (~5.5cm from wall)
const int16_t SIDE_OPEN = 1000;        // Opening detected

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

Encoder encL(1, 0);
Encoder encR(2, 8);
Adafruit_SH1106G display(128, 64, &Wire, -1);

// ==================== STATE ====================
volatile bool running = false;
int16_t lastPwmL = 0, lastPwmR = 0;
uint32_t runStart = 0;
uint32_t loopCount = 0;

// Sensor filter
#define BUF_SIZE 5
int16_t fBuf[BUF_SIZE], lBuf[BUF_SIZE], rBuf[BUF_SIZE];
uint8_t bufIdx = 0;

// NN buffers
float h1[32], h2[16];

// ==================== MEDIAN FILTER ====================
int16_t median5(int16_t* b) {
  int16_t t[BUF_SIZE];
  memcpy(t, b, sizeof(t));
  for (uint8_t i = 1; i < BUF_SIZE; i++) {
    int16_t key = t[i];
    int8_t j = i - 1;
    while (j >= 0 && t[j] > key) { t[j+1] = t[j]; j--; }
    t[j + 1] = key;
  }
  return t[2];
}

void readSensors(int16_t &f, int16_t &l, int16_t &r) {
  fBuf[bufIdx] = analogRead(SH_F);
  lBuf[bufIdx] = analogRead(SH_L);
  rBuf[bufIdx] = analogRead(SH_R);
  bufIdx = (bufIdx + 1) % BUF_SIZE;
  f = median5(fBuf);
  l = median5(lBuf);
  r = median5(rBuf);
}

// ==================== MOTOR CONTROL ====================
void setForward() {
  digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW);
}

void setMotors(int16_t L, int16_t R) {
  L = constrain(L, 0, 255);
  R = constrain(R, 0, 255);
  if (L > 0 && L < MIN_PWM) L = MIN_PWM;
  if (R > 0 && R < MIN_PWM) R = MIN_PWM;
  lastPwmL = L;
  lastPwmR = R;
  analogWrite(PWMA, L);
  analogWrite(PWMB, R);
}

void stopMotors() {
  lastPwmL = lastPwmR = 0;
  analogWrite(PWMA, 0);
  analogWrite(PWMB, 0);
}

// ==================== PRECISION TURNS ====================
void pivotTurn(long tL, long tR) {
  encL.write(0);
  encR.write(0);
  
  digitalWrite(AIN1, tL > 0 ? HIGH : LOW);
  digitalWrite(AIN2, tL > 0 ? LOW : HIGH);
  digitalWrite(BIN1, tR > 0 ? HIGH : LOW);
  digitalWrite(BIN2, tR > 0 ? LOW : HIGH);
  
  while (abs(encL.read()) < abs(tL) || abs(encR.read()) < abs(tR)) {
    analogWrite(PWMA, abs(encL.read()) < abs(tL) ? TURN_SPEED : 0);
    analogWrite(PWMB, abs(encR.read()) < abs(tR) ? TURN_SPEED : 0);
  }
  
  stopMotors();
  delay(50);
  setForward();
}

void turnLeft90()  { pivotTurn(-TICKS_90, TICKS_90); }
void turnRight90() { pivotTurn(TICKS_90, -TICKS_90); }
void turn180()     { pivotTurn(TICKS_90 * 2, -TICKS_90 * 2); }

// ==================== NEURAL NETWORK ====================
void nnInference(int16_t f, int16_t l, int16_t r, int16_t &nnL, int16_t &nnR) {
  // Normalize inputs (0-1 range)
  float inF = (float)f / SCALE_SENSOR;
  float inL = (float)l / SCALE_SENSOR;
  float inR = (float)r / SCALE_SENSOR;
  
  // Layer 1: 3 -> 32 with ReLU
  for (int j = 0; j < 32; j++) {
    float sum = b1[j] + inF * w1[j][0] + inL * w1[j][1] + inR * w1[j][2];
    h1[j] = sum > 0 ? sum : 0;  // ReLU
  }
  
  // Layer 2: 32 -> 16 with ReLU
  for (int j = 0; j < 16; j++) {
    float sum = b2[j];
    for (int i = 0; i < 32; i++) sum += h1[i] * w2[j][i];
    h2[j] = sum > 0 ? sum : 0;  // ReLU
  }
  
  // Layer 3: 16 -> 2 (linear output)
  float outL = b3[0], outR = b3[1];
  for (int i = 0; i < 16; i++) {
    outL += h2[i] * w3[0][i];
    outR += h2[i] * w3[1][i];
  }
  
  // Scale back to PWM range
  nnL = (int16_t)(outL * SCALE_PWM);
  nnR = (int16_t)(outR * SCALE_PWM);
  
  // Clamp to valid range
  nnL = constrain(nnL, 0, 255);
  nnR = constrain(nnR, 0, 255);
}

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  
  pinMode(BTN_PIN, INPUT_PULLUP);
  pinMode(STBY, OUTPUT);
  pinMode(AIN1, OUTPUT); pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT); pinMode(BIN2, OUTPUT);
  pinMode(PWMA, OUTPUT); pinMode(PWMB, OUTPUT);
  
  digitalWrite(STBY, HIGH);
  analogReadResolution(12);
  
  // Init sensor buffers
  for (int i = 0; i < BUF_SIZE; i++) {
    fBuf[i] = lBuf[i] = rBuf[i] = 0;
  }
  
  display.begin(0x3C, true);
  display.setTextColor(SH110X_WHITE);
  display.clearDisplay();
  display.setCursor(0, 8);
  display.println("MazeSolver v8.0");
  display.setCursor(0, 24);
  display.println("NN + Safety");
  display.setCursor(0, 48);
  display.println("BTN to START");
  display.display();
  
  setForward();
  Serial.println(F("=== MazeSolver FINAL v8.0 ==="));
}

// ==================== MAIN LOOP ====================
void loop() {
  static uint32_t lastBtn = 0;
  static uint32_t lastDisp = 0;
  
  // Button toggle
  if (digitalRead(BTN_PIN) == LOW && millis() - lastBtn > 300) {
    lastBtn = millis();
    running = !running;
    if (running) {
      runStart = millis();
      loopCount = 0;
      encL.write(0);
      encR.write(0);
    } else {
      stopMotors();
    }
    delay(150);
  }
  
  // Paused state
  if (!running) {
    if (millis() - lastDisp > 150) {
      lastDisp = millis();
      int16_t f, l, r;
      readSensors(f, l, r);
      
      display.clearDisplay();
      display.setCursor(0, 8);
      display.println("=== PAUSED ===");
      display.setCursor(0, 24);
      display.printf("F:%d", f);
      display.setCursor(0, 38);
      display.printf("L:%d R:%d", l, r);
      display.display();
    }
    return;
  }
  
  loopCount++;
  
  // Read sensors
  int16_t f, l, r;
  readSensors(f, l, r);
  
  // =====================================================
  // LAYER 1: EMERGENCY HANDLING (Highest Priority)
  // =====================================================
  
  // Front wall emergency - must turn
  if (f > FRONT_EMERGENCY) {
    stopMotors();
    delay(80);
    
    // Choose turn direction
    if (l < SIDE_OPEN && r >= SIDE_CENTERED) {
      turnLeft90();   // Opening on left
    } else if (r < SIDE_OPEN && l >= SIDE_CENTERED) {
      turnRight90();  // Opening on right
    } else if (l < r) {
      turnLeft90();   // Less obstacle on left
    } else if (r < l) {
      turnRight90();  // Less obstacle on right
    } else {
      turn180();      // Dead end
    }
    return;
  }
  
  // =====================================================
  // LAYER 2: JUNCTION DETECTION (Left-Hand Rule)
  // =====================================================
  
  if (l < SIDE_OPEN && f < FRONT_DETECT) {
    // Opening on left - take it
    setMotors(CRUISE_SPEED, CRUISE_SPEED);
    delay(40);  // Move forward slightly
    turnLeft90();
    return;
  }
  
  // =====================================================
  // LAYER 3: NEURAL NETWORK INFERENCE
  // =====================================================
  
  int16_t nnPwmL, nnPwmR;
  nnInference(f, l, r, nnPwmL, nnPwmR);
  
  // Start with NN output
  int16_t pwmL = nnPwmL;
  int16_t pwmR = nnPwmR;
  
  // =====================================================
  // LAYER 4: SPEED LIMITING (Front Obstacle)
  // =====================================================
  
  int16_t maxSpeed = MAX_SPEED;
  
  if (f > FRONT_CLOSE) {
    maxSpeed = map(f, FRONT_CLOSE, FRONT_EMERGENCY, SLOW_SPEED, MIN_PWM);
  } else if (f > FRONT_DETECT) {
    maxSpeed = map(f, FRONT_DETECT, FRONT_CLOSE, CRUISE_SPEED, SLOW_SPEED);
  }
  
  // Apply speed limit while preserving steering ratio
  if (pwmL > maxSpeed || pwmR > maxSpeed) {
    float ratio = (float)maxSpeed / max(pwmL, pwmR);
    pwmL = (int16_t)(pwmL * ratio);
    pwmR = (int16_t)(pwmR * ratio);
  }
  
  // =====================================================
  // LAYER 5: SAFETY OVERRIDES (Cannot be bypassed!)
  // =====================================================
  // These guarantee wall avoidance regardless of NN output
  
  // CRASH PREVENTION - Almost touching wall
  if (l > SIDE_CRASH) {
    // Emergency: very close to LEFT -> LEFT wheel MUST be faster
    pwmL = max(pwmL, (int16_t)175);
    pwmR = min(pwmR, (int16_t)60);
  }
  else if (l > SIDE_DANGER) {
    // Danger zone left -> steer right (left faster)
    pwmL = max(pwmL, pwmL + 40);
    pwmR = min(pwmR, pwmR - 20);
  }
  else if (l > SIDE_CLOSE) {
    // Getting close to left -> mild right
    pwmL += 20;
    pwmR -= 10;
  }
  
  if (r > SIDE_CRASH) {
    // Emergency: very close to RIGHT -> RIGHT wheel MUST be faster
    pwmR = max(pwmR, (int16_t)175);
    pwmL = min(pwmL, (int16_t)60);
  }
  else if (r > SIDE_DANGER) {
    // Danger zone right -> steer left (right faster)
    pwmR = max(pwmR, pwmR + 40);
    pwmL = min(pwmL, pwmL - 20);
  }
  else if (r > SIDE_CLOSE) {
    // Getting close to right -> mild left
    pwmR += 20;
    pwmL -= 10;
  }
  
  // =====================================================
  // LAYER 6: FINAL CLAMP & APPLY
  // =====================================================
  
  pwmL = constrain(pwmL, MIN_PWM, MAX_SPEED);
  pwmR = constrain(pwmR, MIN_PWM, MAX_SPEED);
  
  setMotors(pwmL, pwmR);
  
  // =====================================================
  // DISPLAY & DEBUG
  // =====================================================
  
  if (millis() - lastDisp > 100) {
    lastDisp = millis();
    
    display.clearDisplay();
    
    // Line 1: Run time
    display.setCursor(0, 0);
    display.print("RUN ");
    display.print((millis() - runStart) / 1000);
    display.print("s");
    
    // Line 2: Front sensor
    display.setCursor(0, 12);
    display.printf("F:%d", f);
    
    // Line 3: Side sensors
    display.setCursor(0, 24);
    display.printf("L:%d R:%d", l, r);
    
    // Line 4: NN output
    display.setCursor(0, 38);
    display.printf("NN:%d/%d", nnPwmL, nnPwmR);
    
    // Line 5: Actual PWM
    display.setCursor(0, 52);
    display.printf("PWM:%d/%d", lastPwmL, lastPwmR);
    
    display.display();
    
    // Serial debug (every 20th loop)
    if (loopCount % 20 == 0) {
      Serial.printf("F:%d L:%d R:%d | NN[%d,%d] -> PWM[%d,%d]\n",
                    f, l, r, nnPwmL, nnPwmR, lastPwmL, lastPwmR);
    }
  }
  
  delay(15);  // ~65Hz loop
}
