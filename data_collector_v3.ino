/*
 * MazeSolver Data Collector v3.0
 * ================================
 * Calibrated for YOUR robot:
 *   - Robot: 120mm L × 140mm W
 *   - Track: 250mm wide corridors
 *   - Side sensors at center (70mm from edge)
 *   - Clearance: 55mm per side when centered
 * 
 * SENSOR CHARACTERISTICS:
 *   Front (GP2Y0A41SK): 3000 @ 3cm, 2000 @ 2cm, drops < 4cm
 *   Sides (GP2Y0A21): ~4000 when touching
 *   Centered reading: ~1800-2200 each side (55mm + 70mm = 125mm from wall)
 * 
 * CONTROLS:
 *   Short press: Toggle STATIC recording (manually position robot)
 *   Long press (1s): Toggle DRIVE recording (robot moves + records)
 *   Double tap: Next scenario label
 * 
 * AUTO-DETECTION: Scenarios are auto-labeled based on sensor values
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Encoder.h>
#include <SD.h>

// ==================== PINS ====================
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

// ==================== YOUR CALIBRATED THRESHOLDS ====================
// Based on your measurements:
// - Front: 3000 @ 3cm, 2000 @ 2cm
// - Sides: 4000 when touching, ~2000 when centered

const int16_t FRONT_EMERGENCY = 2800;  // ~2.5cm - STOP immediately
const int16_t FRONT_CLOSE     = 2400;  // ~3cm - slow down
const int16_t FRONT_DETECT    = 1800;  // ~4cm - wall ahead
const int16_t FRONT_CLEAR     = 1200;  // No obstacle

const int16_t SIDE_CRASH      = 3800;  // Almost touching!
const int16_t SIDE_DANGER     = 3200;  // ~2cm from wall - EMERGENCY
const int16_t SIDE_CLOSE      = 2600;  // Getting close - need correction
const int16_t SIDE_CENTERED   = 2000;  // Roughly centered (~5.5cm from wall)
const int16_t SIDE_OPEN       = 1000;  // Opening detected

// ==================== SCENARIOS ====================
enum Scenario {
  CENTER = 0,       // Centered in corridor
  CLOSE_LEFT,       // Drifting toward left wall
  CLOSE_RIGHT,      // Drifting toward right wall
  DANGER_LEFT,      // VERY close to left - emergency steer
  DANGER_RIGHT,     // VERY close to right - emergency steer
  DEAD_END,         // Wall in front
  TURN_LEFT,        // Opening on left
  TURN_RIGHT,       // Opening on right
  OPEN_SPACE,       // No walls
  NUM_SCENARIOS
};

const char* scenarioNames[] = {
  "CENTER",
  "CLOSE_L",
  "CLOSE_R",
  "DANGER_L",
  "DANGER_R",
  "DEADEND",
  "TURN_L",
  "TURN_R",
  "OPEN"
};

// IDEAL PWM for each scenario - THIS IS WHAT NN LEARNS
// Format: {left_pwm, right_pwm}
// CRITICAL: When close to LEFT wall, LEFT wheel must be FASTER to turn RIGHT (away from left)
const int16_t idealPWM[NUM_SCENARIOS][2] = {
  {130, 130},   // CENTER: go straight
  {145, 115},   // CLOSE_L: mild right (left faster)
  {115, 145},   // CLOSE_R: mild left (right faster)
  {170, 70},    // DANGER_L: HARD right (left much faster)
  {70, 170},    // DANGER_R: HARD left (right much faster)
  {0, 0},       // DEAD_END: stop
  {70, 130},    // TURN_L: turn left (right faster)
  {130, 70},    // TURN_R: turn right (left faster)
  {140, 140},   // OPEN: cruise fast
};

// ==================== STATE ====================
enum Mode { IDLE, STATIC_REC, DRIVE_REC };
Mode mode = IDLE;
Scenario currentScenario = CENTER;
Scenario detectedScenario = CENTER;

File dataFile;
char filename[24];
uint16_t fileNumber = 0;
uint32_t recordStart = 0;
uint32_t fileStart = 0;  // For 1-min auto-split
uint32_t sampleCount = 0;
uint32_t fileSamples = 0;  // Samples in current file
uint32_t totalSamples = 0;
bool sdReady = false;

// Auto file split every 1 minute
const uint32_t FILE_DURATION_MS = 60000;  // 1 minute per file

// ==================== SENSOR FILTER ====================
#define BUF_SIZE 5
int16_t fBuf[BUF_SIZE], lBuf[BUF_SIZE], rBuf[BUF_SIZE];
uint8_t bufIdx = 0;

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

// ==================== AUTO SCENARIO DETECTION ====================
Scenario detectScenario(int16_t f, int16_t l, int16_t r) {
  // Priority 1: Front blocked
  if (f > FRONT_EMERGENCY) return DEAD_END;
  
  // Priority 2: Danger zones (almost crashing!)
  if (l > SIDE_DANGER) return DANGER_LEFT;
  if (r > SIDE_DANGER) return DANGER_RIGHT;
  
  // Priority 3: Openings for turns
  if (l < SIDE_OPEN && r > SIDE_CENTERED) return TURN_LEFT;
  if (r < SIDE_OPEN && l > SIDE_CENTERED) return TURN_RIGHT;
  
  // Priority 4: Open space
  if (l < SIDE_OPEN && r < SIDE_OPEN && f < FRONT_CLEAR) return OPEN_SPACE;
  
  // Priority 5: Close to one wall
  if (l > SIDE_CLOSE && l > r + 300) return CLOSE_LEFT;
  if (r > SIDE_CLOSE && r > l + 300) return CLOSE_RIGHT;
  
  // Default: centered
  return CENTER;
}

// ==================== MOTOR CONTROL ====================
int16_t curPwmL = 0, curPwmR = 0;
const int16_t BASE_SPD = 110;
const int16_t MIN_PWM = 55;
const int16_t MAX_SPD = 150;
const int16_t TURN_SPD = 110;
const long TICKS_90 = 455;

// PD for centering
float Kp = 0.022f, Kd = 0.010f;
int16_t lastErr = 0;

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
  delay(60);
  setForward();
}

// ==================== SD CARD ====================
void initSD() {
  if (!SD.begin(SD_CS)) {
    Serial.println("SD FAIL!");
    sdReady = false;
    return;
  }
  
  // Find next available file number
  while (fileNumber < 999) {
    sprintf(filename, "maze_%03d.csv", fileNumber);
    if (!SD.exists(filename)) break;
    fileNumber++;
  }
  
  // Count existing samples across all files
  for (int i = 0; i < fileNumber; i++) {
    sprintf(filename, "maze_%03d.csv", i);
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
  Serial.printf("SD Ready. Files: %d, Samples: %lu\n", fileNumber, totalSamples);
}

void startRecording() {
  if (!sdReady) return;
  
  sprintf(filename, "maze_%03d.csv", fileNumber);
  dataFile = SD.open(filename, FILE_WRITE);
  if (dataFile) {
    // CSV header with ideal outputs for supervised learning
    dataFile.println("time_ms,front,left,right,enc_l,enc_r,pwm_l,pwm_r,scenario,ideal_l,ideal_r");
    if (recordStart == 0) recordStart = millis();  // Keep total session time
    fileStart = millis();  // Track this file's start
    fileSamples = 0;
    encL.write(0);
    encR.write(0);
    Serial.printf(">>> Recording: %s\n", filename);
  }
}

// Auto-split: close current file and start new one
void splitFile() {
  if (dataFile) {
    dataFile.flush();
    dataFile.close();
    totalSamples += fileSamples;
    sampleCount += fileSamples;
    Serial.printf("<<< File complete: %lu samples\n", fileSamples);
    fileNumber++;
  }
  startRecording();  // Start new file
}

void stopRecording() {
  if (dataFile) {
    dataFile.flush();
    dataFile.close();
    totalSamples += fileSamples;
    sampleCount += fileSamples;
    Serial.printf("<<< Saved %lu samples. Total: %lu\n", fileSamples, totalSamples);
    fileNumber++;
    recordStart = 0;  // Reset session timer
  }
}

void logSample(int16_t f, int16_t l, int16_t r, Scenario sc) {
  if (!dataFile) return;
  
  // Log: time,f,l,r,encL,encR,pwmL,pwmR,scenario,idealL,idealR
  dataFile.printf("%lu,%d,%d,%d,%ld,%ld,%d,%d,%d,%d,%d\n",
    millis() - recordStart,
    f, l, r,
    encL.read(), encR.read(),
    curPwmL, curPwmR,
    (int)sc,
    idealPWM[sc][0], idealPWM[sc][1]);
  
  fileSamples++;
  if (fileSamples % 50 == 0) dataFile.flush();
}

// ==================== DISPLAY ====================
void updateDisplay(int16_t f, int16_t l, int16_t r) {
  display.clearDisplay();
  display.setTextSize(1);
  
  // Line 1: Mode, time, and file progress
  display.setCursor(0, 0);
  if (mode == IDLE) {
    display.print("IDLE");
  } else if (mode == STATIC_REC) {
    display.print("STAT ");
    display.print((millis() - recordStart) / 1000);
    display.print("s");
  } else {
    display.print("DRIV ");
    display.print((millis() - recordStart) / 1000);
    display.print("s");
  }
  
  // Show file timer (seconds until next split)
  if (mode != IDLE) {
    uint32_t fileElapsed = (millis() - fileStart) / 1000;
    display.setCursor(70, 0);
    display.printf("F%d:%lus", fileNumber, 60 - min(fileElapsed, 60UL));
  }
  
  // Line 2: Sensors + Scenario
  display.setCursor(0, 12);
  display.printf("F:%d", f);
  display.setCursor(64, 12);
  display.printf("SC:%s", scenarioNames[detectedScenario]);
  
  display.setCursor(0, 24);
  display.printf("L:%d R:%d", l, r);
  
  // Line 4: File samples / Total samples
  display.setCursor(0, 36);
  display.printf("File:%lu", fileSamples);
  display.setCursor(64, 36);
  display.printf("Tot:%lu", totalSamples + fileSamples);
  
  // Line 5: Ideal PWM for current scenario
  display.setCursor(0, 48);
  display.printf("Ideal:%d/%d", idealPWM[detectedScenario][0], idealPWM[detectedScenario][1]);
  
  // Line 6: Current PWM (if driving)
  if (mode == DRIVE_REC) {
    display.setCursor(64, 48);
    display.printf("Now:%d/%d", curPwmL, curPwmR);
  }
  
  display.display();
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
  display.setCursor(0, 10);
  display.println("DataCollector v3");
  display.setCursor(0, 26);
  display.println("Your Robot:");
  display.setCursor(0, 38);
  display.println("120x140mm");
  display.display();
  delay(1500);
  
  initSD();
  setForward();
  
  display.clearDisplay();
  display.setCursor(0, 10);
  if (sdReady) {
    display.println("SD Ready!");
    display.setCursor(0, 26);
    display.printf("Files: %d", fileNumber);
    display.setCursor(0, 40);
    display.printf("Samples: %lu", totalSamples);
  } else {
    display.println("NO SD CARD!");
  }
  display.display();
  delay(1500);
  
  Serial.println(F("=== Data Collector v3.0 ==="));
  Serial.println(F("Short press: STATIC mode"));
  Serial.println(F("Long press: DRIVE mode"));
}

// ==================== MAIN LOOP ====================
void loop() {
  static uint32_t lastBtn = 0;
  static uint32_t btnDown = 0;
  static bool wasPressed = false;
  static uint32_t lastLog = 0;
  static uint32_t lastDisp = 0;
  
  bool btnNow = digitalRead(BTN_PIN) == LOW;
  
  // Button handling
  if (btnNow && !wasPressed) {
    btnDown = millis();
    wasPressed = true;
  }
  
  if (!btnNow && wasPressed && millis() - btnDown > 50) {
    uint32_t held = millis() - btnDown;
    wasPressed = false;
    
    if (held > 1000) {
      // Long press: DRIVE mode
      if (mode == DRIVE_REC) {
        stopRecording();
        stopMotors();
        mode = IDLE;
      } else {
        if (mode == STATIC_REC) stopRecording();
        startRecording();
        mode = DRIVE_REC;
      }
    } else if (held > 50) {
      // Short press: STATIC mode
      if (mode == STATIC_REC) {
        stopRecording();
        mode = IDLE;
      } else {
        if (mode == DRIVE_REC) {
          stopRecording();
          stopMotors();
        }
        startRecording();
        mode = STATIC_REC;
      }
    }
    
    delay(100);
  }
  
  // Read sensors
  int16_t f, l, r;
  readSensors(f, l, r);
  
  // Auto-detect scenario
  detectedScenario = detectScenario(f, l, r);
  
  // DRIVE mode with safety
  if (mode == DRIVE_REC) {
    // Calculate base speed + PD centering
    int16_t err = l - r;  // Positive = closer to left
    int16_t dErr = err - lastErr;
    lastErr = err;
    
    // PD correction: positive error -> steer right -> left faster
    int16_t corr = (int16_t)(Kp * err + Kd * dErr);
    corr = constrain(corr, -40, 40);
    
    // Base speed with front slowdown
    int16_t base = BASE_SPD;
    if (f > FRONT_CLOSE) {
      base = map(f, FRONT_CLOSE, FRONT_EMERGENCY, 70, 0);
    } else if (f > FRONT_DETECT) {
      base = map(f, FRONT_DETECT, FRONT_CLOSE, BASE_SPD, 70);
    }
    
    int16_t pwmL = base + corr;
    int16_t pwmR = base - corr;
    
    // SAFETY OVERRIDES - correct steering direction!
    if (l > SIDE_DANGER) {
      // Too close to left - LEFT wheel faster = turn right
      pwmL = max(pwmL, 150);
      pwmR = min(pwmR, 80);
    } else if (l > SIDE_CLOSE) {
      pwmL += 25;
      pwmR -= 10;
    }
    
    if (r > SIDE_DANGER) {
      // Too close to right - RIGHT wheel faster = turn left
      pwmR = max(pwmR, 150);
      pwmL = min(pwmL, 80);
    } else if (r > SIDE_CLOSE) {
      pwmR += 25;
      pwmL -= 10;
    }
    
    // Front emergency
    if (f > FRONT_EMERGENCY) {
      stopMotors();
      delay(100);
      
      // Decide turn
      if (l < SIDE_OPEN && r > SIDE_CENTERED) {
        pivotTurn(-TICKS_90, TICKS_90);  // Left turn
      } else if (r < SIDE_OPEN && l > SIDE_CENTERED) {
        pivotTurn(TICKS_90, -TICKS_90);  // Right turn
      } else if (l < r) {
        pivotTurn(-TICKS_90, TICKS_90);
      } else {
        pivotTurn(TICKS_90, -TICKS_90);
      }
      return;
    }
    
    // Opening detection - left-hand rule
    if (l < SIDE_OPEN && f < FRONT_DETECT) {
      setMotors(BASE_SPD, BASE_SPD);
      delay(40);
      pivotTurn(-TICKS_90, TICKS_90);
      return;
    }
    
    setMotors(pwmL, pwmR);
  } else {
    stopMotors();
  }
  
  // Log data at 50Hz
  if (mode != IDLE && millis() - lastLog >= 20) {
    lastLog = millis();
    logSample(f, l, r, detectedScenario);
    
    // AUTO SPLIT: Every 1 minute, save file and start new one
    if (millis() - fileStart >= FILE_DURATION_MS) {
      splitFile();
      Serial.printf("=== Auto-split at %lu sec ===\n", (millis() - recordStart) / 1000);
    }
  }
  
  // Update display at 8Hz
  if (millis() - lastDisp >= 125) {
    lastDisp = millis();
    updateDisplay(f, l, r);
  }
  
  delay(10);
}
