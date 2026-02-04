/*
 * MazeSolver ULTIMATE Data Collector v2.0
 * =========================================
 * Records sensor data for training a PERFECT wall-avoiding neural network
 * 
 * SCENARIOS TO RECORD (see guide below):
 * 1. CORRIDOR_CENTER - Robot centered between walls
 * 2. CORRIDOR_LEFT   - Robot closer to left wall
 * 3. CORRIDOR_RIGHT  - Robot closer to right wall  
 * 4. DEAD_END        - Wall in front
 * 5. LEFT_TURN       - Opening on left
 * 6. RIGHT_TURN      - Opening on right
 * 7. OPEN_SPACE      - No walls nearby
 * 8. RECOVERY_LEFT   - Almost hitting left wall
 * 9. RECOVERY_RIGHT  - Almost hitting right wall
 * 
 * MODES:
 * - Short press: STATIC recording (hold robot still, move it manually)
 * - Long press (1s): DRIVE mode (robot drives + records)
 * - Double press: Switch scenario label
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Encoder.h>
#include <SD.h>

// ==================== PIN DEFINITIONS ====================
#define PWMA 9
#define PWMB 10
#define AIN1 4
#define AIN2 3
#define BIN1 6
#define BIN2 7
#define STBY 5
#define BTN_PIN 11
#define SD_CS BUILTIN_SDCARD

const uint8_t SH_F = 21, SH_L = 23, SH_R = 22;

Encoder encL(1, 0);
Encoder encR(2, 8);
Adafruit_SH1106G display(128, 64, &Wire, -1);

// ==================== SCENARIOS ====================
enum Scenario {
  CORRIDOR_CENTER = 0,
  CORRIDOR_LEFT,
  CORRIDOR_RIGHT,
  DEAD_END,
  LEFT_TURN,
  RIGHT_TURN,
  OPEN_SPACE,
  RECOVERY_LEFT,
  RECOVERY_RIGHT,
  NUM_SCENARIOS
};

const char* scenarioNames[] = {
  "CENTER",
  "CLOSE_L",
  "CLOSE_R", 
  "DEAD_END",
  "LEFT_TURN",
  "RIGHT_TURN",
  "OPEN",
  "DANGER_L",
  "DANGER_R"
};

// Ideal PWM responses for each scenario (what the NN should learn)
// Format: {pwm_left, pwm_right}
const int16_t idealPWM[NUM_SCENARIOS][2] = {
  {130, 130},   // CENTER: straight
  {150, 110},   // CLOSE_L: steer right (away from left)
  {110, 150},   // CLOSE_R: steer left (away from right)
  {0, 0},       // DEAD_END: stop
  {80, 140},    // LEFT_TURN: turn left
  {140, 80},    // RIGHT_TURN: turn right
  {140, 140},   // OPEN: go fast
  {170, 70},    // DANGER_L: hard right
  {70, 170},    // DANGER_R: hard left
};

// ==================== RECORDING STATE ====================
enum Mode { IDLE, STATIC_REC, DRIVE_REC };
Mode mode = IDLE;
Scenario currentScenario = CORRIDOR_CENTER;

File dataFile;
char filename[24];
uint16_t fileNumber = 0;
uint32_t recordStart = 0;
uint32_t sampleCount = 0;
uint32_t totalSamples = 0;
bool sdReady = false;

// ==================== SENSOR BUFFERS ====================
#define BUF_SIZE 7
int16_t fBuf[BUF_SIZE], lBuf[BUF_SIZE], rBuf[BUF_SIZE];
uint8_t bufIdx = 0;

// ==================== MOTOR STATE ====================
int16_t curPwmL = 0, curPwmR = 0;
const int16_t BASE_SPD = 120;
const int16_t MIN_PWM = 55;
const int16_t MAX_SPD = 160;
const int16_t TURN_SPD = 120;
const long TICKS_90 = 455;

// Thresholds for auto-labeling
const int16_t FRONT_STOP = 2600;
const int16_t FRONT_SLOW = 2000;
const int16_t SIDE_DANGER = 3000;
const int16_t SIDE_CLOSE = 2500;
const int16_t SIDE_OPEN = 1000;

// PD Control
float Kp = 0.025f, Kd = 0.012f;
int16_t lastErr = 0;

// ==================== MEDIAN FILTER ====================
int16_t median7(int16_t* b) {
  int16_t t[BUF_SIZE];
  memcpy(t, b, sizeof(t));
  // Insertion sort
  for (uint8_t i = 1; i < BUF_SIZE; i++) {
    int16_t key = t[i];
    int8_t j = i - 1;
    while (j >= 0 && t[j] > key) {
      t[j + 1] = t[j];
      j--;
    }
    t[j + 1] = key;
  }
  return t[BUF_SIZE / 2];
}

void readSensors(int16_t &f, int16_t &l, int16_t &r) {
  fBuf[bufIdx] = analogRead(SH_F);
  lBuf[bufIdx] = analogRead(SH_L);
  rBuf[bufIdx] = analogRead(SH_R);
  bufIdx = (bufIdx + 1) % BUF_SIZE;
  f = median7(fBuf);
  l = median7(lBuf);
  r = median7(rBuf);
}

// ==================== AUTO SCENARIO DETECTION ====================
Scenario detectScenario(int16_t f, int16_t l, int16_t r) {
  // Front blocked
  if (f > FRONT_STOP) return DEAD_END;
  
  // Danger zones
  if (l > SIDE_DANGER) return RECOVERY_LEFT;
  if (r > SIDE_DANGER) return RECOVERY_RIGHT;
  
  // Openings (for turns)
  if (l < SIDE_OPEN && r > SIDE_CLOSE) return LEFT_TURN;
  if (r < SIDE_OPEN && l > SIDE_CLOSE) return RIGHT_TURN;
  
  // Open space
  if (l < SIDE_OPEN && r < SIDE_OPEN && f < 800) return OPEN_SPACE;
  
  // Close to walls
  if (l > SIDE_CLOSE && r < SIDE_CLOSE) return CORRIDOR_LEFT;
  if (r > SIDE_CLOSE && l < SIDE_CLOSE) return CORRIDOR_RIGHT;
  
  // Balanced
  return CORRIDOR_CENTER;
}

// ==================== SD CARD ====================
void initSD() {
  if (!SD.begin(SD_CS)) {
    Serial.println("SD FAIL");
    sdReady = false;
    return;
  }
  
  // Find next file
  while (fileNumber < 999) {
    sprintf(filename, "data_%03d.csv", fileNumber);
    if (!SD.exists(filename)) break;
    fileNumber++;
  }
  
  // Count existing samples
  for (int i = 0; i < fileNumber; i++) {
    sprintf(filename, "data_%03d.csv", i);
    File f = SD.open(filename, FILE_READ);
    if (f) {
      while (f.available()) {
        if (f.read() == '\n') totalSamples++;
      }
      f.close();
    }
  }
  totalSamples = max(0, (int)totalSamples - fileNumber); // Subtract headers
  
  sdReady = true;
  Serial.printf("SD OK. Files: %d, Samples: %lu\n", fileNumber, totalSamples);
}

void startRecording() {
  if (!sdReady) return;
  
  sprintf(filename, "data_%03d.csv", fileNumber);
  dataFile = SD.open(filename, FILE_WRITE);
  if (dataFile) {
    // Header with scenario and ideal PWM for supervised learning
    dataFile.println("time_ms,front,left,right,enc_l,enc_r,pwm_l,pwm_r,scenario,ideal_l,ideal_r");
    recordStart = millis();
    sampleCount = 0;
    encL.write(0);
    encR.write(0);
    Serial.printf("Recording: %s\n", filename);
  }
}

void stopRecording() {
  if (dataFile) {
    dataFile.flush();
    dataFile.close();
    totalSamples += sampleCount;
    Serial.printf("Saved %lu samples. Total: %lu\n", sampleCount, totalSamples);
    fileNumber++;
  }
}

void logSample(int16_t f, int16_t l, int16_t r, Scenario sc) {
  if (!dataFile) return;
  
  dataFile.print(millis() - recordStart);
  dataFile.print(',');
  dataFile.print(f);
  dataFile.print(',');
  dataFile.print(l);
  dataFile.print(',');
  dataFile.print(r);
  dataFile.print(',');
  dataFile.print(encL.read());
  dataFile.print(',');
  dataFile.print(encR.read());
  dataFile.print(',');
  dataFile.print(curPwmL);
  dataFile.print(',');
  dataFile.print(curPwmR);
  dataFile.print(',');
  dataFile.print((int)sc);
  dataFile.print(',');
  dataFile.print(idealPWM[sc][0]);
  dataFile.print(',');
  dataFile.println(idealPWM[sc][1]);
  
  sampleCount++;
  if (sampleCount % 50 == 0) dataFile.flush();
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
  curPwmL = L;
  curPwmR = R;
  analogWrite(PWMA, L);
  analogWrite(PWMB, R);
}

void stopMotors() {
  curPwmL = curPwmR = 0;
  analogWrite(PWMA, 0);
  analogWrite(PWMB, 0);
}

void pivotTurn(long tL, long tR) {
  encL.write(0);
  encR.write(0);
  digitalWrite(AIN1, tL > 0 ? HIGH : LOW);
  digitalWrite(AIN2, tL > 0 ? LOW : HIGH);
  digitalWrite(BIN1, tR > 0 ? HIGH : LOW);
  digitalWrite(BIN2, tR > 0 ? LOW : HIGH);
  
  while (abs(encL.read()) < abs(tL) || abs(encR.read()) < abs(tR)) {
    analogWrite(PWMA, abs(encL.read()) < abs(tL) ? TURN_SPD : 0);
    analogWrite(PWMB, abs(encR.read()) < abs(tR) ? TURN_SPD : 0);
  }
  stopMotors();
  delay(80);
  setForward();
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
  display.setCursor(0, 0);
  display.println("DATA COLLECTOR v2.0");
  display.setCursor(0, 16);
  display.println("Short: Static Rec");
  display.setCursor(0, 28);
  display.println("Long:  Drive Rec");
  display.setCursor(0, 40);
  display.println("Double: Change Mode");
  display.display();
  
  initSD();
  setForward();
  delay(1000);
}

// ==================== MAIN LOOP ====================
void loop() {
  static uint32_t lastBtn = 0;
  static uint32_t btnDown = 0;
  static uint8_t pressCount = 0;
  static uint32_t lastPress = 0;
  static uint32_t lastSample = 0;
  static uint32_t lastDisplay = 0;
  
  bool btnPressed = (digitalRead(BTN_PIN) == LOW);
  uint32_t now = millis();
  
  // Button state machine
  if (btnPressed && btnDown == 0) {
    btnDown = now;
  }
  
  if (!btnPressed && btnDown > 0) {
    uint32_t duration = now - btnDown;
    btnDown = 0;
    
    if (mode != IDLE) {
      // Any press while recording = stop
      stopMotors();
      stopRecording();
      mode = IDLE;
    } else {
      if (duration > 1000) {
        // Long press = Drive + Record
        mode = DRIVE_REC;
        startRecording();
      } else {
        pressCount++;
        lastPress = now;
      }
    }
  }
  
  // Check for double-press (change scenario)
  if (pressCount > 0 && now - lastPress > 400) {
    if (pressCount >= 2) {
      currentScenario = (Scenario)((currentScenario + 1) % NUM_SCENARIOS);
    } else {
      // Single press = Static record
      mode = STATIC_REC;
      startRecording();
    }
    pressCount = 0;
  }
  
  // Read sensors
  int16_t f, l, r;
  readSensors(f, l, r);
  
  // Auto-detect scenario
  Scenario detected = detectScenario(f, l, r);
  
  if (mode == IDLE) {
    // Show status
    if (now - lastDisplay > 100) {
      lastDisplay = now;
      display.clearDisplay();
      display.setCursor(0, 0);
      display.printf("IDLE  Total:%lu", totalSamples);
      
      display.setCursor(0, 14);
      display.printf("F:%d", f);
      
      display.setCursor(0, 26);
      display.printf("L:%d R:%d", l, r);
      
      display.setCursor(0, 40);
      display.printf("Detected: %s", scenarioNames[detected]);
      
      display.setCursor(0, 54);
      display.printf("Manual: %s", scenarioNames[currentScenario]);
      
      display.display();
    }
    return;
  }
  
  // Recording modes
  if (now - lastSample >= 20) {  // 50Hz
    lastSample = now;
    
    Scenario logScenario = (mode == STATIC_REC) ? currentScenario : detected;
    
    if (mode == DRIVE_REC) {
      // Autonomous driving with good behavior
      
      // Emergency front stop
      if (f > FRONT_STOP) {
        stopMotors();
        logSample(f, l, r, DEAD_END);
        delay(100);
        if (l < r) pivotTurn(-TICKS_90, TICKS_90);
        else pivotTurn(TICKS_90, -TICKS_90);
        return;
      }
      
      // Left opening - take it
      if (l < SIDE_OPEN && f < FRONT_SLOW) {
        setMotors(BASE_SPD, BASE_SPD);
        delay(50);
        logSample(f, l, r, LEFT_TURN);
        pivotTurn(-TICKS_90, TICKS_90);
        return;
      }
      
      // PD centering
      int16_t err = l - r;
      int16_t dErr = err - lastErr;
      lastErr = err;
      int16_t corr = (int16_t)(Kp * err + Kd * dErr);
      corr = constrain(corr, -50, 50);
      
      // Speed based on front distance
      int16_t spd = (f > FRONT_SLOW) ? 
        map(f, FRONT_SLOW, FRONT_STOP, MAX_SPD, MIN_PWM + 20) : MAX_SPD;
      
      int16_t pwmL = spd - corr;
      int16_t pwmR = spd + corr;
      
      // Wall avoidance (CORRECT direction!)
      if (l > SIDE_DANGER) { pwmL += 50; pwmR -= 25; }
      if (r > SIDE_DANGER) { pwmL -= 25; pwmR += 50; }
      if (l > SIDE_CLOSE && l <= SIDE_DANGER) { pwmL += 20; pwmR -= 10; }
      if (r > SIDE_CLOSE && r <= SIDE_DANGER) { pwmL -= 10; pwmR += 20; }
      
      setMotors(pwmL, pwmR);
    }
    
    // Log with ideal PWM targets
    logSample(f, l, r, logScenario);
  }
  
  // Display update
  if (now - lastDisplay > 150) {
    lastDisplay = now;
    display.clearDisplay();
    
    display.setCursor(0, 0);
    display.print(mode == STATIC_REC ? "STATIC " : "DRIVE  ");
    display.print((now - recordStart) / 1000);
    display.print("s");
    
    display.setCursor(0, 12);
    display.printf("Samples: %lu", sampleCount);
    
    display.setCursor(0, 26);
    display.printf("F:%d L:%d R:%d", f, l, r);
    
    display.setCursor(0, 40);
    display.printf("PWM: %d/%d", curPwmL, curPwmR);
    
    display.setCursor(0, 54);
    display.printf("Scene: %s", scenarioNames[detected]);
    
    display.display();
  }
}
