/*--------------------------------------------------
  Teensy 4.1 | 16-Channel Line Follower - LOOP/CURVE OPTIMIZED
  Designed for complex curved tracks with loops
  
  Features:
  - Adaptive speed based on curve detection
  - Loop detection (12+ sensors active)
  - Sharp curve detection (4+ edge sensors)
  - Smooth acceleration/deceleration
  - Button: Press = Calibrate + Auto Start
--------------------------------------------------*/

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

// ==================== CONFIGURATION ====================
bool isBlackLine = true;
bool calibrated = false;
bool runEnabled = false;

// Adaptive PID - adjusted for curves
float Kp = 0.08;
float Kd = 1.2;
float Ki = 0.0001;

// Speed profiles for different track conditions
#define MAX_SPEED 240           // Straight sections
#define CRUISE_SPEED 200        // Normal cruising
#define CURVE_SPEED 140         // Sharp curves
#define TIGHT_CURVE_SPEED 100   // Very tight curves/loops
#define START_SPEED 120         // Initial acceleration
#define CAL_SPEED 180           // Calibration spin speed

// Acceleration parameters
#define ACCEL_STEP 5            // Speed increase per loop
#define DECEL_STEP 8            // Speed decrease per loop (faster braking)

// Curve detection thresholds
#define SHARP_CURVE_THRESHOLD 4   // How many edge sensors trigger = sharp curve
#define CURVE_THRESHOLD 3         // Moderate curve

// 0-7 RIGHT | 8-15 LEFT (weighted for position calculation)
int sensorWeight[16] = {
 -8,-7,-6,-5,-4,-3,-2,-1,  // Right sensors (more weight on outer)
  1, 2, 3, 4, 5, 6, 7, 8   // Left sensors
};

int minVal[16], maxVal[16], threshold[16];
int sensorVal[16], sensorBin[16];

int P, D, I, prevError;
int lsp, rsp;
bool onLine;

int baseSpeed = 180;                // Base cruising speed (reduced for loops)
int currentSpeed = START_SPEED;     // Actual motor speed
int targetSpeed = 180;              // Dynamic target based on curves

// Track analysis
int activeSensors = 0;
int leftActiveSensors = 0;
int rightActiveSensors = 0;
bool sharpCurveDetected = false;
bool loopDetected = false;

unsigned long startTime;
unsigned long lastOnLineTime = 0;
unsigned long lapStartTime = 0;

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

  Serial.println("\n========================================");
  Serial.println("  16-CH LINE FOLLOWER - LOOP OPTIMIZED");
  Serial.println("========================================");
  Serial.println("Press Button = CALIBRATE + AUTO START\n");
}

// ==================== HELPER FUNCTIONS ====================
void motorsForward() {
  // Start both motors forward at start speed
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, HIGH);
  digitalWrite(BIN2, LOW);
  analogWrite(PWMA, START_SPEED);
  analogWrite(PWMB, START_SPEED);
  Serial.print("motorsForward() - PWM A:");
  Serial.print(START_SPEED);
  Serial.print(" B:");
  Serial.println(START_SPEED);
}

// ==================== MAIN LOOP ====================
void loop() {
  handleButton();

  if (!runEnabled) return;

  readLine();  // This now calls analyzeTrack() internally
  
  // Adaptive speed control based on track conditions
  updateTargetSpeed();
  smoothSpeedAdjust();

  if (onLine) {
    lastOnLineTime = millis();
    lineFollow();
  } else {
    handleLineLoss();
  }
}

// ==================== BUTTON HANDLER ====================
void handleButton() {
  static bool lastBtnState = HIGH;
  bool btnState = digitalRead(BTN_PIN);

  // Detect button press (falling edge)
  if (lastBtnState == HIGH && btnState == LOW && !runEnabled) {
    Serial.println("\n[BTN] Button Pressed → CALIBRATION");
    
    // Run calibration
    autoCalibrate();
    calibrated = true;
    
    // Automatically start line following after calibration
    Serial.println("\n[AUTO] Starting line following...\n");
    runEnabled = true;
    currentSpeed = START_SPEED;
    targetSpeed = CRUISE_SPEED;
    lapStartTime = millis();
    lastOnLineTime = millis();
    I = 0;
    prevError = 0;
    
    // Start moving forward immediately
    Serial.print("Starting motors at PWM: ");
    Serial.println(START_SPEED);
    motorsForward();
    
    Serial.println("Motors running!");
  }
  
  // Stop if button pressed during run
  if (lastBtnState == HIGH && btnState == LOW && runEnabled) {
    Serial.println("\n[BTN] STOPPED by user");
    runEnabled = false;
    motorLeft(0);
    motorRight(0);
  }

  lastBtnState = btnState;
}

// ==================== CALIBRATION ====================
void autoCalibrate() {
  Serial.println("=== AUTO CALIBRATION START ===");

  // Initialize min/max
  for (int i = 0; i < 16; i++) {
    minVal[i] = sensorRead(i);
    maxVal[i] = sensorRead(i);
  }

  // Gentle ramp-up to avoid current spike
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
    
    // Progress indicator
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
  Serial.println("Sensor Thresholds:");

  for (int i = 0; i < 16; i++) {
    threshold[i] = (minVal[i] + maxVal[i]) / 2;
    Serial.print("S");
    if (i < 10) Serial.print("0");
    Serial.print(i);
    Serial.print(":");
    Serial.print(threshold[i]);
    if ((i + 1) % 4 == 0) Serial.println();
    else Serial.print("  ");
  }
  Serial.println("============================\n");
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
      if (i < 8) rightActiveSensors++;  // Right sensors
      else leftActiveSensors++;         // Left sensors
    }
  }
  
  // Analyze track conditions
  analyzeTrack();
}

// ==================== TRACK ANALYSIS ====================
void analyzeTrack() {
  // Detect sharp curves/loops
  sharpCurveDetected = false;
  loopDetected = false;
  
  // Sharp right curve: many right sensors active
  if (rightActiveSensors >= SHARP_CURVE_THRESHOLD) {
    sharpCurveDetected = true;
  }
  
  // Sharp left curve: many left sensors active
  if (leftActiveSensors >= SHARP_CURVE_THRESHOLD) {
    sharpCurveDetected = true;
  }
  
  // Loop detection: almost all sensors active (crossing over itself)
  if (activeSensors >= 12) {
    loopDetected = true;
  }
}

// ==================== ADAPTIVE SPEED CONTROL ====================
void updateTargetSpeed() {
  // Determine target speed based on track conditions
  if (loopDetected) {
    targetSpeed = TIGHT_CURVE_SPEED;
  } else if (sharpCurveDetected) {
    targetSpeed = CURVE_SPEED;
  } else if (activeSensors >= CURVE_THRESHOLD) {
    targetSpeed = CRUISE_SPEED;
  } else if (activeSensors <= 2) {
    // Straight line or slight curve - full speed
    targetSpeed = MAX_SPEED;
  } else {
    targetSpeed = CRUISE_SPEED;
  }
}

void smoothSpeedAdjust() {
  // Smooth acceleration/deceleration
  if (currentSpeed < targetSpeed) {
    currentSpeed = min(targetSpeed, currentSpeed + ACCEL_STEP);
  } else if (currentSpeed > targetSpeed) {
    currentSpeed = max(targetSpeed, currentSpeed - DECEL_STEP);
  }
  
  // Safety: never go below minimum
  currentSpeed = constrain(currentSpeed, 80, MAX_SPEED);
}

// ==================== LINE FOLLOWING ====================
void lineFollow() {
  int error = 0;
  int weightedSum = 0;
  int totalValue = 0;

  // Calculate weighted position error
  for (int i = 0; i < 16; i++) {
    if (sensorBin[i]) {
      weightedSum += sensorWeight[i] * sensorVal[i];
      totalValue += sensorVal[i];
    }
  }

  if (totalValue == 0) return;
  error = weightedSum / totalValue;

  // PID calculation
  P = error;
  I += error;
  D = error - prevError;
  prevError = error;

  // Anti-windup for integral
  I = constrain(I, -5000, 5000);

  int pid = (Kp * P) + (Ki * I) + (Kd * D);

  // Apply PID correction to current speed
  lsp = currentSpeed + pid;
  rsp = currentSpeed - pid;

  lsp = constrain(lsp, -255, 255);
  rsp = constrain(rsp, -255, 255);

  motorLeft(lsp);
  motorRight(rsp);

  // -------- DEBUG OUTPUT ----------
  static uint32_t dbgT = 0;
  if (millis() - dbgT > 150) {
    dbgT = millis();
    
    Serial.print("ERR:");
    Serial.print(error);
    Serial.print(" | Sensors:");
    Serial.print(activeSensors);
    Serial.print(" L:");
    Serial.print(leftActiveSensors);
    Serial.print(" R:");
    Serial.print(rightActiveSensors);
    Serial.print(" | SPD:");
    Serial.print(currentSpeed);
    Serial.print("/");
    Serial.print(targetSpeed);
    Serial.print(" | PWM L:");
    Serial.print(lsp);
    Serial.print(" R:");
    Serial.print(rsp);
    
    if (loopDetected) Serial.print(" [LOOP!]");
    else if (sharpCurveDetected) Serial.print(" [CURVE]");
    
    Serial.println();
  }
}

// ==================== LINE LOSS RECOVERY ====================
void handleLineLoss() {
  unsigned long lostTime = millis() - lastOnLineTime;
  
  if (lostTime < 300) {
    // Just lost line - continue with last known direction
    if (prevError > 0) {
      motorLeft(-constrain(currentSpeed * 0.6, 80, 200));
      motorRight(constrain(currentSpeed * 0.9, 80, 200));
    } else if (prevError < 0) {
      motorLeft(constrain(currentSpeed * 0.9, 80, 200));
      motorRight(-constrain(currentSpeed * 0.6, 80, 200));
    } else {
      // No previous error - go straight
      motorLeft(START_SPEED);
      motorRight(START_SPEED);
    }
  } else if (lostTime < 1500) {
    // Still lost - sharper turn
    if (prevError > 0) {
      motorLeft(-180);
      motorRight(200);
    } else if (prevError < 0) {
      motorLeft(200);
      motorRight(-180);
    } else {
      // Spin to search
      motorLeft(150);
      motorRight(-150);
    }
    
    // Debug output
    static uint32_t lossDbg = 0;
    if (millis() - lossDbg > 200) {
      lossDbg = millis();
      Serial.print("SEARCHING... Lost for: ");
      Serial.print(lostTime);
      Serial.println("ms");
    }
  } else {
    // Lost for too long - stop
    motorLeft(0);
    motorRight(0);
    Serial.println("LINE LOST - STOPPED");
    runEnabled = false;
  }
}

// ==================== MOTOR CONTROL ====================
void motorLeft(int spd) {
  static uint32_t lastDbg = 0;
  static int lastSpd = 0;
  
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
  
  // Debug motor commands occasionally
  if (runEnabled && (millis() - lastDbg > 500 || abs(spd - lastSpd) > 30)) {
    lastDbg = millis();
    lastSpd = spd;
    Serial.print("[L:");
    Serial.print(spd);
    Serial.print("] ");
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

// ==================== SENSOR MULTIPLEXER ====================
int sensorRead(int ch) {
  digitalWrite(S0, ch & 0x01);
  digitalWrite(S1, ch & 0x02);
  digitalWrite(S2, ch & 0x04);
  digitalWrite(S3, ch & 0x08);
  delayMicroseconds(5);
  return analogRead(SIGNAL_PIN);
}
