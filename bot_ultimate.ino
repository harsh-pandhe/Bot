/*
 * MazeSolver ULTIMATE v7.0 - ZERO WALL HITS
 * ==========================================
 * - Neural Network with CORRECT steering logic
 * - Hard safety overrides that CANNOT be bypassed
 * - Encoder-based precision turns
 * - Multi-layer wall avoidance
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Encoder.h>
#include "weights.h"

// ==================== TUNE THESE ====================
const long TICKS_90 = 455;    // Encoder ticks for 90-degree turn

// Speed limits
const int16_t MAX_SPEED = 160;   // Maximum forward speed
const int16_t CRUISE_SPEED = 130; // Normal cruising speed
const int16_t SLOW_SPEED = 80;   // Approaching obstacle
const int16_t MIN_PWM = 55;      // Minimum to overcome friction
const int16_t TURN_SPEED = 120;  // Pivot turn speed

// Safety thresholds (ADC values - higher = closer)
const int16_t FRONT_EMERGENCY = 2700;  // HARD STOP
const int16_t FRONT_CLOSE = 2200;      // Start slowing
const int16_t FRONT_DETECT = 1800;     // Wall ahead

const int16_t SIDE_CRASH = 3200;       // EMERGENCY - almost touching
const int16_t SIDE_DANGER = 2900;      // Strong correction needed
const int16_t SIDE_CLOSE = 2500;       // Mild correction
const int16_t SIDE_WALL = 2000;        // Wall present
const int16_t SIDE_OPEN = 900;         // Opening detected

// ==================== PINS ====================
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
int16_t lastError = 0;
int16_t lastPwmL = 0, lastPwmR = 0;
uint32_t runStart = 0;
uint32_t loopCount = 0;

// Sensor buffers
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
    while (j >= 0 && t[j] > key) {
      t[j + 1] = t[j];
      j--;
    }
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
  lastPwmL = 0;
  lastPwmR = 0;
  analogWrite(PWMA, 0);
  analogWrite(PWMB, 0);
}

// ==================== PRECISION TURNS ====================
void pivotTurn(long ticksL, long ticksR) {
  encL.write(0);
  encR.write(0);
  
  digitalWrite(AIN1, ticksL > 0 ? HIGH : LOW);
  digitalWrite(AIN2, ticksL > 0 ? LOW : HIGH);
  digitalWrite(BIN1, ticksR > 0 ? HIGH : LOW);
  digitalWrite(BIN2, ticksR > 0 ? LOW : HIGH);
  
  while (abs(encL.read()) < abs(ticksL) || abs(encR.read()) < abs(ticksR)) {
    analogWrite(PWMA, abs(encL.read()) < abs(ticksL) ? TURN_SPEED : 0);
    analogWrite(PWMB, abs(encR.read()) < abs(ticksR) ? TURN_SPEED : 0);
  }
  
  stopMotors();
  delay(60);
  setForward();
}

void turnLeft90()  { pivotTurn(-TICKS_90, TICKS_90); }
void turnRight90() { pivotTurn(TICKS_90, -TICKS_90); }
void turn180()     { pivotTurn(TICKS_90 * 2, -TICKS_90 * 2); }

// ==================== NEURAL NETWORK ====================
void nnInference(int16_t f, int16_t l, int16_t r, int16_t &nnL, int16_t &nnR) {
  // Normalize inputs
  float inF = (float)f / SCALE_SENSOR;
  float inL = (float)l / SCALE_SENSOR;
  float inR = (float)r / SCALE_SENSOR;
  
  // Layer 1: 3 -> 32 with ReLU
  for (int j = 0; j < 32; j++) {
    float sum = b1[j] + inF * w1[j][0] + inL * w1[j][1] + inR * w1[j][2];
    h1[j] = sum > 0 ? sum : 0;
  }
  
  // Layer 2: 32 -> 16 with ReLU
  for (int j = 0; j < 16; j++) {
    float sum = b2[j];
    for (int i = 0; i < 32; i++) sum += h1[i] * w2[j][i];
    h2[j] = sum > 0 ? sum : 0;
  }
  
  // Layer 3: 16 -> 2 linear
  float outL = b3[0], outR = b3[1];
  for (int i = 0; i < 16; i++) {
    outL += h2[i] * w3[0][i];
    outR += h2[i] * w3[1][i];
  }
  
  nnL = (int16_t)(outL * SCALE_PWM);
  nnR = (int16_t)(outR * SCALE_PWM);
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
  
  memset(fBuf, 0, sizeof(fBuf));
  memset(lBuf, 0, sizeof(lBuf));
  memset(rBuf, 0, sizeof(rBuf));
  
  display.begin(0x3C, true);
  display.setTextColor(SH110X_WHITE);
  display.clearDisplay();
  display.setCursor(0, 8);
  display.println("ULTIMATE v7.0");
  display.setCursor(0, 24);
  display.println("ZERO WALL HITS");
  display.setCursor(0, 48);
  display.println("BTN to START");
  display.display();
  
  setForward();
  Serial.println(F("=== MazeSolver ULTIMATE v7.0 ==="));
}

// ==================== MAIN LOOP ====================
void loop() {
  static uint32_t lastBtn = 0;
  static uint32_t lastDisplay = 0;
  
  // Button handling
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
  
  if (!running) {
    if (millis() - lastDisplay > 150) {
      lastDisplay = millis();
      int16_t f, l, r;
      readSensors(f, l, r);
      
      display.clearDisplay();
      display.setCursor(0, 10);
      display.println("=== PAUSED ===");
      display.setCursor(0, 26);
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
  
  // ===============================================
  // LAYER 1: EMERGENCY STOPS (HIGHEST PRIORITY)
  // ===============================================
  
  // Front emergency - wall too close
  if (f > FRONT_EMERGENCY) {
    stopMotors();
    delay(100);
    
    // Decide turn direction
    if (l < SIDE_OPEN && r >= SIDE_WALL) {
      turnLeft90();
    } else if (r < SIDE_OPEN && l >= SIDE_WALL) {
      turnRight90();
    } else if (l < r) {
      turnLeft90();
    } else if (r < l) {
      turnRight90();
    } else {
      // Dead end
      turn180();
    }
    return;
  }
  
  // ===============================================
  // LAYER 2: JUNCTION HANDLING (LEFT-HAND RULE)
  // ===============================================
  
  if (l < SIDE_OPEN && f < FRONT_DETECT) {
    // Opening on left - take it
    setMotors(CRUISE_SPEED, CRUISE_SPEED);
    delay(50);
    turnLeft90();
    return;
  }
  
  // ===============================================
  // LAYER 3: NEURAL NETWORK SUGGESTION
  // ===============================================
  
  int16_t nnPwmL, nnPwmR;
  nnInference(f, l, r, nnPwmL, nnPwmR);
  
  // ===============================================
  // LAYER 4: PD CENTERING
  // ===============================================
  
  int16_t error = l - r;  // Positive = closer to left
  int16_t dError = error - lastError;
  lastError = error;
  
  // PD correction
  // Positive error -> steer right -> increase left PWM, decrease right PWM
  int16_t pdCorr = (int16_t)(0.018f * error + 0.008f * dError);
  pdCorr = constrain(pdCorr, -40, 40);
  
  // ===============================================
  // LAYER 5: SPEED CALCULATION
  // ===============================================
  
  int16_t baseSpeed;
  
  if (f > FRONT_CLOSE) {
    baseSpeed = map(f, FRONT_CLOSE, FRONT_EMERGENCY, SLOW_SPEED, MIN_PWM);
  } else if (f > FRONT_DETECT) {
    baseSpeed = map(f, FRONT_DETECT, FRONT_CLOSE, CRUISE_SPEED, SLOW_SPEED);
  } else {
    baseSpeed = CRUISE_SPEED;
  }
  
  // ===============================================
  // LAYER 6: COMBINE NN + PD
  // ===============================================
  
  // Blend NN output with base speed
  // NN provides steering direction, we control magnitude
  int16_t nnSteer = (nnPwmL - nnPwmR) / 5;
  
  int16_t pwmL = baseSpeed + pdCorr + nnSteer;
  int16_t pwmR = baseSpeed - pdCorr - nnSteer;
  
  // ===============================================
  // LAYER 7: HARD SAFETY OVERRIDES (CANNOT BE BYPASSED!)
  // ===============================================
  
  // CRASH AVOIDANCE - this is the LAST LINE OF DEFENSE
  // When very close to wall, FORCE steering away regardless of NN/PD
  
  if (l > SIDE_CRASH) {
    // Almost hitting left wall - FORCE hard right
    pwmL = max(pwmL, 170);  // Left wheel MUST be fast
    pwmR = min(pwmR, 70);   // Right wheel MUST be slow
  }
  else if (l > SIDE_DANGER) {
    // Danger zone left - strong right correction
    pwmL = max(pwmL, pwmL + 45);
    pwmR = min(pwmR, pwmR - 20);
  }
  else if (l > SIDE_CLOSE) {
    // Getting close to left - mild right
    pwmL += 20;
    pwmR -= 8;
  }
  
  if (r > SIDE_CRASH) {
    // Almost hitting right wall - FORCE hard left
    pwmL = min(pwmL, 70);   // Left wheel MUST be slow
    pwmR = max(pwmR, 170);  // Right wheel MUST be fast
  }
  else if (r > SIDE_DANGER) {
    // Danger zone right - strong left correction
    pwmL = min(pwmL, pwmL - 20);
    pwmR = max(pwmR, pwmR + 45);
  }
  else if (r > SIDE_CLOSE) {
    // Getting close to right - mild left
    pwmL -= 8;
    pwmR += 20;
  }
  
  // ===============================================
  // LAYER 8: FINAL CONSTRAIN & APPLY
  // ===============================================
  
  pwmL = constrain(pwmL, MIN_PWM, MAX_SPEED);
  pwmR = constrain(pwmR, MIN_PWM, MAX_SPEED);
  
  setMotors(pwmL, pwmR);
  
  // ===============================================
  // DISPLAY & DEBUG
  // ===============================================
  
  if (millis() - lastDisplay > 120) {
    lastDisplay = millis();
    
    display.clearDisplay();
    display.setCursor(0, 0);
    display.print("RUN ");
    display.print((millis() - runStart) / 1000);
    display.print("s");
    
    display.setCursor(0, 12);
    display.printf("F:%d", f);
    
    display.setCursor(0, 24);
    display.printf("L:%d R:%d", l, r);
    
    display.setCursor(0, 38);
    display.printf("PWM %d/%d", lastPwmL, lastPwmR);
    
    display.setCursor(0, 52);
    display.printf("NN:%d/%d", nnPwmL, nnPwmR);
    
    display.display();
    
    // Serial debug
    if (loopCount % 25 == 0) {
      Serial.printf("F:%d L:%d R:%d -> PWM[%d,%d] NN[%d,%d]\n",
                    f, l, r, lastPwmL, lastPwmR, nnPwmL, nnPwmR);
    }
  }
  
  delay(20);  // 50Hz
}
