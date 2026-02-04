/*
 * MazeSolver PD v10.0 - Enhanced Smooth Control
 * ===============================================
 * Pure PD control with wall following + smooth control improvements
 * Robot: 120mm x 140mm, Track: 250mm
 * 
 * NEW in v10.0:
 *   - Exponential moving average filter (on top of median)
 *   - Speed ramping (prevents sudden motor changes)
 *   - Dead-band (eliminates micro-oscillations)
 *   - Smooth turns (ramp up/down during pivots)
 *   - Improved safety overrides with smooth transitions
 * 
 * Based on v9.0 with v21.0 smooth control techniques
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

// NEW: Smooth control parameters
const int16_t DEADBAND = 100;     // Ignore errors smaller than this (eliminates oscillations)
const int16_t RAMP_RATE = 8;      // Max PWM change per loop (prevents sudden jumps)
const float FILTER_ALPHA = 0.3f;  // Exponential filter weight (0.3 = smooth but responsive)

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
int16_t currentPwmL = 0, currentPwmR = 0;  // NEW: Current actual PWM (for ramping)
int16_t lastError = 0;
uint32_t runStart = 0;
uint32_t loopCount = 0;

// Sensor filter buffers
#define BUF_SIZE 5
int16_t fBuf[BUF_SIZE], lBuf[BUF_SIZE], rBuf[BUF_SIZE];
uint8_t bufIdx = 0;

// NEW: Exponential moving average (smooths out median filter output)
float fSmooth = 0.0f, lSmooth = 0.0f, rSmooth = 0.0f;
bool firstRead = true;

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

// NEW: Two-stage filtering (median + exponential moving average)
void readSensors(int16_t &f, int16_t &l, int16_t &r) {
  // Stage 1: Median filter (removes spikes)
  fBuf[bufIdx] = analogRead(SH_F);
  lBuf[bufIdx] = analogRead(SH_L);
  rBuf[bufIdx] = analogRead(SH_R);
  bufIdx = (bufIdx + 1) % BUF_SIZE;
  
  int16_t fMedian = median5(fBuf);
  int16_t lMedian = median5(lBuf);
  int16_t rMedian = median5(rBuf);
  
  // Stage 2: Exponential moving average (smooths trends)
  if (firstRead) {
    fSmooth = fMedian;
    lSmooth = lMedian;
    rSmooth = rMedian;
    firstRead = false;
  } else {
    fSmooth = FILTER_ALPHA * fMedian + (1.0f - FILTER_ALPHA) * fSmooth;
    lSmooth = FILTER_ALPHA * lMedian + (1.0f - FILTER_ALPHA) * lSmooth;
    rSmooth = FILTER_ALPHA * rMedian + (1.0f - FILTER_ALPHA) * rSmooth;
  }
  
  f = (int16_t)fSmooth;
  l = (int16_t)lSmooth;
  r = (int16_t)rSmooth;
}

// ==================== MOTOR CONTROL ====================
void setForward() {
  digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW);
}

// NEW: Smooth motor control with ramping
void setMotors(int16_t targetL, int16_t targetR) {
  // Clamp targets
  targetL = constrain(targetL, 0, 255);
  targetR = constrain(targetR, 0, 255);
  
  // Apply minimum PWM threshold
  if (targetL > 0 && targetL < MIN_PWM) targetL = MIN_PWM;
  if (targetR > 0 && targetR < MIN_PWM) targetR = MIN_PWM;
  
  // Ramp towards target (prevents sudden changes)
  if (currentPwmL < targetL) {
    currentPwmL = min(currentPwmL + RAMP_RATE, targetL);
  } else if (currentPwmL > targetL) {
    currentPwmL = max(currentPwmL - RAMP_RATE, targetL);
  }
  
  if (currentPwmR < targetR) {
    currentPwmR = min(currentPwmR + RAMP_RATE, targetR);
  } else if (currentPwmR > targetR) {
    currentPwmR = max(currentPwmR - RAMP_RATE, targetR);
  }
  
  lastPwmL = currentPwmL;
  lastPwmR = currentPwmR;
  
  analogWrite(PWMA, currentPwmL);
  analogWrite(PWMB, currentPwmR);
}

void stopMotors() {
  lastPwmL = lastPwmR = 0;
  currentPwmL = currentPwmR = 0;
  analogWrite(PWMA, 0);
  analogWrite(PWMB, 0);
}

// ==================== SMOOTH PRECISION TURNS ====================
// NEW: Turns with smooth acceleration and deceleration
void pivotTurn(long tL, long tR) {
  encL.write(0);
  encR.write(0);
  
  digitalWrite(AIN1, tL > 0 ? HIGH : LOW);
  digitalWrite(AIN2, tL > 0 ? LOW : HIGH);
  digitalWrite(BIN1, tR > 0 ? HIGH : LOW);
  digitalWrite(BIN2, tR > 0 ? LOW : HIGH);
  
  const long accelTicks = abs(tL) / 4;  // Accelerate over first 25%
  const long decelTicks = abs(tL) / 4;  // Decelerate over last 25%
  
  while (abs(encL.read()) < abs(tL) || abs(encR.read()) < abs(tR)) {
    long posL = abs(encL.read());
    long posR = abs(encR.read());
    
    // Smooth speed profile for each wheel
    int16_t speedL = 0, speedR = 0;
    
    if (posL < abs(tL)) {
      if (posL < accelTicks) {
        // Accelerate
        speedL = map(posL, 0, accelTicks, MIN_PWM, TURN_SPEED);
      } else if (posL > abs(tL) - decelTicks) {
        // Decelerate
        speedL = map(posL, abs(tL) - decelTicks, abs(tL), TURN_SPEED, MIN_PWM);
      } else {
        // Full speed
        speedL = TURN_SPEED;
      }
    }
    
    if (posR < abs(tR)) {
      if (posR < accelTicks) {
        speedR = map(posR, 0, accelTicks, MIN_PWM, TURN_SPEED);
      } else if (posR > abs(tR) - decelTicks) {
        speedR = map(posR, abs(tR) - decelTicks, abs(tR), TURN_SPEED, MIN_PWM);
      } else {
        speedR = TURN_SPEED;
      }
    }
    
    analogWrite(PWMA, speedL);
    analogWrite(PWMB, speedR);
  }
  
  stopMotors();
  delay(50);
  setForward();
}

void turnLeft90()  { pivotTurn(-TICKS_90, TICKS_90); }
void turnRight90() { pivotTurn(TICKS_90, -TICKS_90); }
void turn180()     { pivotTurn(TICKS_90 * 2, -TICKS_90 * 2); }

// NEW: Move forward with smooth acceleration/deceleration
void moveForward(long ticks) {
  encL.write(0);
  encR.write(0);
  setForward();
  
  const long accelTicks = ticks / 3;
  const long decelTicks = ticks / 3;
  
  while (abs(encL.read()) < ticks && abs(encR.read()) < ticks) {
    long pos = (abs(encL.read()) + abs(encR.read())) / 2;
    
    int16_t speed;
    if (pos < accelTicks) {
      speed = map(pos, 0, accelTicks, MIN_PWM, BASE_SPEED);
    } else if (pos > ticks - decelTicks) {
      speed = map(pos, ticks - decelTicks, ticks, BASE_SPEED, MIN_PWM);
    } else {
      speed = BASE_SPEED;
    }
    
    setMotors(speed, speed);
    delay(5);
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
  display.println("MazeSolver v10.0");
  display.setCursor(0, 24);
  display.println("Smooth PD Control");
  display.setCursor(0, 48);
  display.println("BTN to START");
  display.display();
  
  setForward();
  Serial.println(F("=== MazeSolver PD v10.0 ==="));
  Serial.println(F("NEW: Smooth ramping + filtering"));
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
      firstRead = true;  // Reset filter
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
  
  // Read sensors (now with dual-stage filtering)
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
    moveForward(TICKS_FORWARD);  // Move forward a bit (now smooth!)
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
  // STEP 4: PD CENTERING CONTROL with DEAD-BAND
  // =====================================================
  
  // Error = left - right
  // Positive error = closer to left wall
  int16_t error = l - r;
  
  // NEW: Apply dead-band (ignore tiny errors that cause oscillations)
  if (abs(error) < DEADBAND) {
    error = 0;
  }
  
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
  // STEP 5: SAFETY OVERRIDES (Smoother transitions)
  // =====================================================
  
  // NEW: More gradual safety corrections (less jarring)
  if (l > SIDE_CRASH) {
    // Almost touching left wall - FORCE hard right
    pwmL = 170;
    pwmR = 60;
  } else if (l > SIDE_DANGER) {
    // Danger zone left - strong right correction (smoother)
    int16_t dangerCorrection = map(l, SIDE_DANGER, SIDE_CRASH, 40, 60);
    pwmL = baseSpeed + dangerCorrection;
    pwmR = baseSpeed - dangerCorrection;
  }
  
  if (r > SIDE_CRASH) {
    // Almost touching right wall - FORCE hard left
    pwmL = 60;
    pwmR = 170;
  } else if (r > SIDE_DANGER) {
    // Danger zone right - strong left correction (smoother)
    int16_t dangerCorrection = map(r, SIDE_DANGER, SIDE_CRASH, 40, 60);
    pwmL = baseSpeed - dangerCorrection;
    pwmR = baseSpeed + dangerCorrection;
  }
  
  // =====================================================
  // STEP 6: CLAMP AND APPLY (with ramping)
  // =====================================================
  
  pwmL = constrain(pwmL, MIN_PWM, MAX_SPEED);
  pwmR = constrain(pwmR, MIN_PWM, MAX_SPEED);
  
  setMotors(pwmL, pwmR);  // Now with smooth ramping!
  
  // =====================================================
  // DISPLAY & DEBUG
  // =====================================================
  
  if (millis() - lastDisp > 100) {
    lastDisp = millis();
    
    display.clearDisplay();
    
    // Line 1: Run time
    display.setCursor(0, 0);
    display.print("v10 ");
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
