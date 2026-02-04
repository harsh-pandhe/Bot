/*
 * MazeSolver PD v9.0 - No Neural Network
 * =======================================
 * Pure PD control with wall following
 * Robot: 120mm x 140mm, Track: 250mm
 * 
 * Features:
 *   - Wall centering using PD control
 *   - Distance maintenance from walls
 *   - Left-hand rule navigation
 *   - Smooth turning with encoders
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Encoder.h>

// ==================== TUNABLE PARAMETERS ====================
const long TICKS_90 = 455;        // Encoder ticks for 90° turn
const long TICKS_FORWARD = 200;   // Ticks to move forward before turn

// Speed settings
const int16_t BASE_SPEED = 120;   // Normal forward speed
const int16_t MAX_SPEED = 160;    // Maximum speed
const int16_t SLOW_SPEED = 80;    // Approaching wall
const int16_t MIN_PWM = 55;       // Minimum to move
const int16_t TURN_SPEED = 110;   // Pivot turn speed

// PD Control gains
const float Kp = 0.035f;          // Proportional gain for centering
const float Kd = 0.015f;          // Derivative gain for damping

// Thresholds (calibrated for your sensors)
const int16_t FRONT_EMERGENCY = 2800;  // ~2.5cm - STOP & turn
const int16_t FRONT_CLOSE = 2200;      // ~3cm - slow down
const int16_t FRONT_DETECT = 1600;     // ~4cm - wall ahead

const int16_t SIDE_CRASH = 3600;       // Almost touching - EMERGENCY
const int16_t SIDE_DANGER = 3200;      // ~2cm - strong correction
const int16_t SIDE_CLOSE = 2600;       // Getting close
const int16_t SIDE_TARGET = 2000;      // Ideal centered distance
const int16_t SIDE_OPEN = 900;         // Opening detected (for turns)

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
int16_t lastError = 0;
uint32_t runStart = 0;
uint32_t loopCount = 0;

// Sensor filter
#define BUF_SIZE 5
int16_t fBuf[BUF_SIZE], lBuf[BUF_SIZE], rBuf[BUF_SIZE];
uint8_t bufIdx = 0;

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

// Move forward by encoder ticks
void moveForward(long ticks) {
  encL.write(0);
  encR.write(0);
  setForward();
  
  while (abs(encL.read()) < ticks && abs(encR.read()) < ticks) {
    setMotors(BASE_SPEED, BASE_SPEED);
  }
  stopMotors();
  delay(30);
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
  display.println("MazeSolver v9.0");
  display.setCursor(0, 24);
  display.println("PD Control");
  display.setCursor(0, 48);
  display.println("BTN to START");
  display.display();
  
  setForward();
  Serial.println(F("=== MazeSolver PD v9.0 ==="));
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
      lastError = 0;
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
  // STEP 1: FRONT WALL CHECK - Must turn if blocked
  // =====================================================
  
  if (f > FRONT_EMERGENCY) {
    stopMotors();
    delay(80);
    
    // Decide turn direction based on side sensors
    if (l < SIDE_OPEN && r >= SIDE_TARGET) {
      // Opening on left, wall on right -> turn left
      turnLeft90();
    } else if (r < SIDE_OPEN && l >= SIDE_TARGET) {
      // Opening on right, wall on left -> turn right
      turnRight90();
    } else if (l < r) {
      // Less obstacle on left
      turnLeft90();
    } else if (r < l) {
      // Less obstacle on right
      turnRight90();
    } else {
      // Dead end - turn around
      turn180();
    }
    lastError = 0;
    return;
  }
  
  // =====================================================
  // STEP 2: LEFT OPENING CHECK (Left-Hand Rule)
  // =====================================================
  
  if (l < SIDE_OPEN && f < FRONT_DETECT) {
    // Opening on left detected - take it
    moveForward(TICKS_FORWARD);  // Move forward a bit
    turnLeft90();
    lastError = 0;
    return;
  }
  
  // =====================================================
  // STEP 3: CALCULATE BASE SPEED (Slow near front wall)
  // =====================================================
  
  int16_t baseSpeed = BASE_SPEED;
  
  if (f > FRONT_CLOSE) {
    // Very close to front wall - slow down significantly
    baseSpeed = map(f, FRONT_CLOSE, FRONT_EMERGENCY, SLOW_SPEED, MIN_PWM);
  } else if (f > FRONT_DETECT) {
    // Approaching front wall - gradual slowdown
    baseSpeed = map(f, FRONT_DETECT, FRONT_CLOSE, BASE_SPEED, SLOW_SPEED);
  }
  
  // =====================================================
  // STEP 4: PD CENTERING CONTROL
  // =====================================================
  
  // Error = left - right
  // Positive error = closer to left wall
  // To correct: make left wheel faster (turn right, away from left)
  
  int16_t error = l - r;
  int16_t dError = error - lastError;
  lastError = error;
  
  // PD calculation
  int16_t correction = (int16_t)(Kp * error + Kd * dError);
  correction = constrain(correction, -60, 60);
  
  // Apply correction
  // Positive correction -> left wheel faster -> turns right
  int16_t pwmL = baseSpeed + correction;
  int16_t pwmR = baseSpeed - correction;
  
  // =====================================================
  // STEP 5: SAFETY OVERRIDES (Hard limits)
  // =====================================================
  
  // Emergency crash prevention - override PD if too close
  if (l > SIDE_CRASH) {
    // Almost touching left wall - FORCE hard right
    pwmL = 170;
    pwmR = 60;
  } else if (l > SIDE_DANGER) {
    // Danger zone left - strong right correction
    pwmL = max(pwmL, baseSpeed + 50);
    pwmR = min(pwmR, baseSpeed - 30);
  }
  
  if (r > SIDE_CRASH) {
    // Almost touching right wall - FORCE hard left
    pwmL = 60;
    pwmR = 170;
  } else if (r > SIDE_DANGER) {
    // Danger zone right - strong left correction
    pwmL = min(pwmL, baseSpeed - 30);
    pwmR = max(pwmR, baseSpeed + 50);
  }
  
  // =====================================================
  // STEP 6: CLAMP AND APPLY
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
    
    // Line 4: Error and correction
    display.setCursor(0, 38);
    display.printf("Err:%d Cor:%d", error, correction);
    
    // Line 5: PWM output
    display.setCursor(0, 52);
    display.printf("PWM:%d/%d", lastPwmL, lastPwmR);
    
    display.display();
    
    // Serial debug
    if (loopCount % 15 == 0) {
      Serial.printf("F:%d L:%d R:%d | Err:%d -> PWM[%d,%d]\n",
                    f, l, r, error, lastPwmL, lastPwmR);
    }
  }
  
  delay(12);  // ~80Hz loop
}
