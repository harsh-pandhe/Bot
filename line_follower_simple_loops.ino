/*--------------------------------------------------
  Teensy 4.1 | 16-Channel Line Follower - LOOP OPTIMIZED
  With Encoder Support for Square Maze Navigation
  Button control: Press to calibrate + auto-start
--------------------------------------------------*/

#include <Encoder.h>

#define AIN1 4
#define AIN2 3
#define BIN1 6
#define BIN2 7
#define PWMA 9
#define PWMB 10
#define STBY 5

#define S0 14
#define S1 15
#define S2 16
#define S3 17
#define SIGNAL_PIN 20   // A6

#define BTN_PIN 11
#define NUM_SENSORS 16

// Encoder pins
#define ENC_LA 1
#define ENC_LB 0
#define ENC_RA 2
#define ENC_RB 8

bool isBlackLine = true;
bool calibrated = false;
bool runEnabled = false;

// PID
float Kp = 0.06;
float Kd = 1.0;
float Ki = 0.0;

int baseSpeed = 220;
int currentSpeed = 120;

#define START_SPEED 120
#define CAL_SPEED 180
#define ACCEL_STEP 4

// ==================== ENCODER SETUP ====================
Encoder encL(ENC_LA, ENC_LB);
Encoder encR(ENC_RA, ENC_RB);

#define TICKS_PER_CM 19.5        // Calibrate for your wheels
#define GRID_CELL_CM 30          // Square grid cell size (adjust for your track)
#define TICKS_PER_CELL (GRID_CELL_CM * TICKS_PER_CM)
#define TURN_THRESHOLD 8         // Sensors active = possible intersection

long encLeft = 0;
long encRight = 0;
long lastCellDistance = 0;
int cellCount = 0;
bool atIntersection = false;

// ==================== LOOP DETECTION ====================
#define LOOP_THRESHOLD 12        // 12+ sensors = loop crossing
#define SHARP_CURVE_THRESHOLD 4  // 4+ edge sensors = sharp curve
#define LOOP_SPEED 100           // Slow speed for loops
#define CURVE_SPEED 140          // Moderate speed for curves

int activeSensors = 0;
int leftActiveSensors = 0;
int rightActiveSensors = 0;
bool loopDetected = false;
bool sharpCurveDetected = false;
int targetSpeed = 220;

// Sensor weights: 0-7 RIGHT | 8-15 LEFT
int sensorWeight[16] = {
 -7,-6,-5,-4,-3,-2,-1, 0,
  0, 1, 2, 3, 4, 5, 6, 7
};

int minVal[16], maxVal[16], threshold[16];
int sensorVal[16], sensorBin[16];

int P, D, I, prevError;
int lsp, rsp;
bool onLine;

unsigned long startTime;

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);
  pinMode(PWMA, OUTPUT);
  pinMode(PWMB, OUTPUT);
  pinMode(STBY, OUTPUT);
  digitalWrite(STBY, HIGH);

  pinMode(S0, OUTPUT);
  pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);

  pinMode(BTN_PIN, INPUT_PULLUP);

  analogReadResolution(10);
  analogReadAveraging(4);

  Serial.println("\n=== LINE FOLLOWER WITH LOOP DETECTION ===");
  Serial.println("Press Button = CALIBRATE + START\n");
}

// ==================== MAIN LOOP ====================
void loop() {
  handleButton();

  if (!runEnabled) return;

  // Read encoders
  updateEncoders();
  
  readLine();
  
  // ========== SQUARE MAZE DETECTION ==========
  detectIntersections();
  
  // ========== LOOP DETECTION ==========
  detectLoopAndCurves();
  adaptiveSpeedControl();

  if (onLine) {
    lineFollowWithEncoders();
  } else {
    // Line lost recovery
    if (prevError > 0) {
      motorLeft(-constrain(baseSpeed * 0.55, 80, 255));
      motorRight(constrain(baseSpeed, 80, 255));
    } else {
      motorLeft(constrain(baseSpeed, 80, 255));
      motorRight(-constrain(baseSpeed * 0.55, 80, 255));
    }
  }
}

// ==================== BUTTON ====================
void handleButton() {
  static bool lastBtn = HIGH;
  bool btn = digitalRead(BTN_PIN);

  if (lastBtn == HIGH && btn == LOW) {
    if (!runEnabled) {
      Serial.println("[BTN] Starting calibration...");
      autoCalibrate();
      
      Serial.println("\n[AUTO] Starting run!");
      runEnabled = true;
      currentSpeed = START_SPEED;
      targetSpeed = baseSpeed;
      resetEncoders();
      I = 0;
      prevError = 0;
    } else {
      Serial.println("[BTN] STOPPED");
      runEnabled = false;
      motorLeft(0);
      motorRight(0);
    }
  }

  lastBtn = btn;
}

// ==================== ENCODER FUNCTIONS ====================
void updateEncoders() {
  encLeft = encL.read();
  encRight = encR.read();
  
  // Track cells traveled
  long avgDistance = (abs(encLeft) + abs(encRight)) / 2;
  if (avgDistance - lastCellDistance > TICKS_PER_CELL) {
    cellCount++;
    lastCellDistance = avgDistance;
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

// ==================== CALIBRATION ====================
void autoCalibrate() {
  Serial.println("=== CALIBRATION START ===");

  for (int i = 0; i < 16; i++) {
    minVal[i] = sensorRead(i);
    maxVal[i] = sensorRead(i);
  }

  // Ramp up
  for (int s = 0; s <= CAL_SPEED; s += 30) {
    motorLeft(s);
    motorRight(-s);
    delay(80);
  }

  startTime = millis();
  while (millis() - startTime < 5000) {
    motorLeft(CAL_SPEED);
    motorRight(-CAL_SPEED);

    for (int i = 0; i < 16; i++) {
      int v = sensorRead(i);
      if (v < minVal[i]) minVal[i] = v;
      if (v > maxVal[i]) maxVal[i] = v;
    }
    
    if ((millis() - startTime) % 1000 < 50) {
      Serial.print(".");
    }
  }

  // Ramp down
  for (int s = CAL_SPEED; s >= 0; s -= 30) {
    motorLeft(s);
    motorRight(-s);
    delay(60);
  }

  motorLeft(0);
  motorRight(0);

  Serial.println("\nCalibration DONE");
  Serial.println("Thresholds:");

  for (int i = 0; i < 16; i++) {
    threshold[i] = (minVal[i] + maxVal[i]) / 2;
    Serial.print("S");
    Serial.print(i);
    Serial.print(":");
    Serial.print(threshold[i]);
    Serial.print("  ");
    if ((i + 1) % 4 == 0) Serial.println();
  }
  Serial.println("=========================\n");
}

// ==================== SENSOR READING ====================
void readLine() {
  onLine = false;
  activeSensors = 0;
  leftActiveSensors = 0;
  rightActiveSensors = 0;

  for (int i = 0; i < 16; i++) {
    if (isBlackLine)
      sensorVal[i] = map(sensorRead(i), minVal[i], maxVal[i], 0, 1000);
    else
      sensorVal[i] = map(sensorRead(i), minVal[i], maxVal[i], 1000, 0);

    sensorVal[i] = constrain(sensorVal[i], 0, 1000);
    sensorBin[i] = sensorVal[i] > 500;

    if (sensorBin[i]) {
      onLine = true;
      activeSensors++;
      if (i < 8) rightActiveSensors++;  // Right sensors (0-7)
      else leftActiveSensors++;         // Left sensors (8-15)
    }
  }
}

// ==================== INTERSECTION DETECTION ====================
void detectIntersections() {
  // Square maze: T-junction or cross when many sensors active
  if (activeSensors >= TURN_THRESHOLD) {
    if (!atIntersection) {
      atIntersection = true;
      // Slow down at intersections
      targetSpeed = LOOP_SPEED;
    }
  } else {
    atIntersection = false;
  }
}

// ==================== LOOP DETECTION ====================
void detectLoopAndCurves() {
  loopDetected = false;
  sharpCurveDetected = false;
  
  // Loop detection: almost all sensors see line (crossing point)
  if (activeSensors >= LOOP_THRESHOLD) {
    loopDetected = true;
  }
  // Sharp curve: many edge sensors active on one side
  else if (rightActiveSensors >= SHARP_CURVE_THRESHOLD || 
           leftActiveSensors >= SHARP_CURVE_THRESHOLD) {
    sharpCurveDetected = true;
  }
}

// ==================== ADAPTIVE SPEED CONTROL ====================
void adaptiveSpeedControl() {
  // Determine target speed based on track conditions
  if (loopDetected) {
    targetSpeed = LOOP_SPEED;  // Slow down for loops
  } else if (sharpCurveDetected) {
    targetSpeed = CURVE_SPEED;  // Moderate speed for sharp curves
  } else if (activeSensors <= 2) {
    targetSpeed = baseSpeed;  // Full speed on straight/slight curves
  } else {
    targetSpeed = baseSpeed - 40;  // Slightly slower for normal curves
  }
  
  // Smooth acceleration/deceleration
  if (currentSpeed < targetSpeed) {
    currentSpeed = min(targetSpeed, currentSpeed + ACCEL_STEP);
  } else if (currentSpeed > targetSpeed) {
    currentSpeed = max(targetSpeed, currentSpeed - (ACCEL_STEP * 2));  // Brake faster
  }
}

// ==================== LINE FOLLOWING WITH ENCODERS ====================
void lineFollowWithEncoders() {
  int error = 0;
  int active = 0;

  for (int i = 0; i < 16; i++) {
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

  // Calculate base motor speeds
  lsp = currentSpeed + pid;
  rsp = currentSpeed - pid;

  // ========== ENCODER BALANCE CORRECTION ==========
  // Keep motors balanced for straight lines in squares
  long encDiff = encLeft - encRight;
  int balanceCorrection = constrain(encDiff * 0.05, -15, 15);
  
  // Apply correction to keep robot straight
  lsp -= balanceCorrection;
  rsp += balanceCorrection;

  lsp = constrain(lsp, -255, 255);
  rsp = constrain(rsp, -255, 255);

  motorLeft(lsp);
  motorRight(rsp);

  // DEBUG
  static uint32_t dbgT = 0;
  if (millis() - dbgT > 150) {
    dbgT = millis();
    Serial.print("ERR:");
    Serial.print(error);
    Serial.print(" | Sens:");
    Serial.print(activeSensors);
    Serial.print(" L:");
    Serial.print(leftActiveSensors);
    Serial.print(" R:");
    Serial.print(rightActiveSensors);
    Serial.print(" | Enc L:");
    Serial.print(encLeft);
    Serial.print(" R:");
    Serial.print(encRight);
    Serial.print(" Cell:");
    Serial.print(cellCount);
    Serial.print(" | SPD:");
    Serial.print(currentSpeed);
    Serial.print("/");
    Serial.print(targetSpeed);
    Serial.print(" PWM:");
    Serial.print(lsp);
    Serial.print("/");
    Serial.print(rsp);
    
    if (atIntersection) Serial.print(" [CROSS]");
    else if (loopDetected) Serial.print(" [LOOP!]");
    else if (sharpCurveDetected) Serial.print(" [CURVE]");
    
    Serial.println();
  }
}

// ==================== MOTOR CONTROL ====================
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

// ==================== SENSOR MUX ====================
int sensorRead(int ch) {
  digitalWrite(S0, ch & 0x01);
  digitalWrite(S1, ch & 0x02);
  digitalWrite(S2, ch & 0x04);
  digitalWrite(S3, ch & 0x08);
  delayMicroseconds(3);
  return analogRead(SIGNAL_PIN);
}
