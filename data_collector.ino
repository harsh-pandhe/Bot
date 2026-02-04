/*
 * MazeSolver Data Collector v1.0
 * Records Sharp IR + Encoder data to SD card for Neural Network training
 * 
 * Creates CSV files: run_001.csv, run_002.csv, etc.
 * Each file = ~1 minute of data (~3000 samples at 50Hz)
 * 
 * Hardware: Teensy 4.1 (built-in SD slot)
 */

#include <SD.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

// ==================== PIN DEFINITIONS ====================
// Motors
#define PWMA 9
#define PWMB 10
#define AIN1 4
#define AIN2 3
#define BIN1 6
#define BIN2 7
#define STBY 5

// Sensors
const uint8_t SH_F = 21, SH_L = 23, SH_R = 22;

// Encoders (adjust pins for your setup)
#define ENC_L_A 2
#define ENC_L_B 3
#define ENC_R_A 4
#define ENC_R_B 5

// Button
#define BTN_PIN 11

// SD Card (Teensy 4.1 built-in)
#define SD_CS BUILTIN_SDCARD

// OLED
Adafruit_SH1106G display(128, 64, &Wire, -1);

// ==================== RECORDING SETTINGS ====================
const uint32_t RECORD_DURATION_MS = 60000;  // 1 minute per file
const uint16_t SAMPLE_INTERVAL_MS = 20;     // 50Hz sampling
const uint16_t SAMPLES_PER_FILE = 3000;     // ~60s at 50Hz

// ==================== DATA STRUCTURES ====================
struct SampleData {
  uint32_t timestamp;
  int16_t frontIR;
  int16_t leftIR;
  int16_t rightIR;
  int16_t encLeft;
  int16_t encRight;
  int16_t pwmLeft;
  int16_t pwmRight;
  uint8_t action;  // 0=FWD, 1=LEFT, 2=RIGHT, 3=STOP, 4=CRASH
};

// ==================== ENCODER VARIABLES ====================
volatile int32_t encCountL = 0;
volatile int32_t encCountR = 0;
int32_t lastEncL = 0, lastEncR = 0;

// ==================== STATE ====================
enum Mode { IDLE, RECORDING, DRIVING };
Mode mode = IDLE;

uint16_t fileNumber = 0;
char filename[20];
File dataFile;
uint32_t recordStartTime = 0;
uint32_t sampleCount = 0;
uint32_t lastSampleTime = 0;

// Sensor buffers
#define BUF_SIZE 5
int16_t fBuf[BUF_SIZE], lBuf[BUF_SIZE], rBuf[BUF_SIZE];
uint8_t bufIdx = 0;
int16_t sF = 0, sL = 0, sR = 0;

// Motor state
int16_t curPwmL = 0, curPwmR = 0;
uint8_t curAction = 0;

// Motion parameters
const int16_t BASE_SPD = 120;
const int16_t MIN_SPD = 50;
const int16_t MAX_SPD = 160;
const int16_t TURN_MS = 450;
const int16_t TURN_PWM = 140;

// Thresholds
const int16_t FRONT_STOP = 2400;
const int16_t FRONT_SLOW = 1800;
const int16_t SIDE_CRASH = 3000;
const int16_t SIDE_WALL = 2000;
const int16_t SIDE_OPEN = 900;

// PID
float Kp = 0.02f, Kd = 0.01f;
int16_t lastErr = 0;

// ==================== ENCODER ISRs ====================
void encLeftISR() {
  if (digitalRead(ENC_L_B)) encCountL++;
  else encCountL--;
}

void encRightISR() {
  if (digitalRead(ENC_R_B)) encCountR++;
  else encCountR--;
}

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

void readSensors() {
  fBuf[bufIdx] = analogRead(SH_F);
  lBuf[bufIdx] = analogRead(SH_L);
  rBuf[bufIdx] = analogRead(SH_R);
  bufIdx = (bufIdx + 1) % BUF_SIZE;
  sF = median5(fBuf);
  sL = median5(lBuf);
  sR = median5(rBuf);
}

// ==================== MOTOR CONTROL ====================
void setMotors(int16_t L, int16_t R) {
  curPwmL = constrain(L, 0, 255);
  curPwmR = constrain(R, 0, 255);
  if (curPwmL > 0 && curPwmL < MIN_SPD) curPwmL = MIN_SPD;
  if (curPwmR > 0 && curPwmR < MIN_SPD) curPwmR = MIN_SPD;
  analogWrite(PWMA, curPwmL);
  analogWrite(PWMB, curPwmR);
}

void stopMotors() {
  curPwmL = 0;
  curPwmR = 0;
  analogWrite(PWMA, 0);
  analogWrite(PWMB, 0);
}

void setForward() {
  digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW);
}

void turnLeft90() {
  curAction = 1;
  stopMotors();
  delay(50);
  digitalWrite(AIN1, LOW); digitalWrite(AIN2, HIGH);
  digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW);
  analogWrite(PWMA, TURN_PWM);
  analogWrite(PWMB, TURN_PWM);
  delay(TURN_MS);
  stopMotors();
  delay(30);
  setForward();
}

void turnRight90() {
  curAction = 2;
  stopMotors();
  delay(50);
  digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, LOW); digitalWrite(BIN2, HIGH);
  analogWrite(PWMA, TURN_PWM);
  analogWrite(PWMB, TURN_PWM);
  delay(TURN_MS);
  stopMotors();
  delay(30);
  setForward();
}

// ==================== SD CARD FUNCTIONS ====================
bool initSD() {
  if (!SD.begin(SD_CS)) {
    Serial.println("SD init failed!");
    return false;
  }
  Serial.println("SD init OK");
  
  // Find next file number
  while (fileNumber < 999) {
    sprintf(filename, "run_%03d.csv", fileNumber);
    if (!SD.exists(filename)) break;
    fileNumber++;
  }
  Serial.print("Next file: ");
  Serial.println(filename);
  return true;
}

bool startNewFile() {
  sprintf(filename, "run_%03d.csv", fileNumber);
  dataFile = SD.open(filename, FILE_WRITE);
  if (!dataFile) {
    Serial.print("Failed to create: ");
    Serial.println(filename);
    return false;
  }
  
  // Write CSV header
  dataFile.println("time_ms,front,left,right,enc_l,enc_r,pwm_l,pwm_r,action");
  dataFile.flush();
  
  recordStartTime = millis();
  sampleCount = 0;
  lastEncL = encCountL;
  lastEncR = encCountR;
  
  Serial.print("Recording to: ");
  Serial.println(filename);
  return true;
}

void writeSample() {
  int16_t deltaEncL = encCountL - lastEncL;
  int16_t deltaEncR = encCountR - lastEncR;
  lastEncL = encCountL;
  lastEncR = encCountR;
  
  // Write CSV line
  dataFile.print(millis() - recordStartTime);
  dataFile.print(',');
  dataFile.print(sF);
  dataFile.print(',');
  dataFile.print(sL);
  dataFile.print(',');
  dataFile.print(sR);
  dataFile.print(',');
  dataFile.print(deltaEncL);
  dataFile.print(',');
  dataFile.print(deltaEncR);
  dataFile.print(',');
  dataFile.print(curPwmL);
  dataFile.print(',');
  dataFile.print(curPwmR);
  dataFile.print(',');
  dataFile.println(curAction);
  
  sampleCount++;
  
  // Flush every 100 samples
  if (sampleCount % 100 == 0) {
    dataFile.flush();
  }
}

void closeFile() {
  if (dataFile) {
    dataFile.flush();
    dataFile.close();
    Serial.print("Saved: ");
    Serial.print(filename);
    Serial.print(" (");
    Serial.print(sampleCount);
    Serial.println(" samples)");
    fileNumber++;
  }
}

// ==================== DISPLAY ====================
void updateDisplay() {
  display.clearDisplay();
  display.setCursor(0, 0);
  
  if (mode == IDLE) {
    display.println("=== DATA COLLECTOR ===");
    display.setCursor(0, 16);
    display.print("Files: ");
    display.println(fileNumber);
    display.setCursor(0, 32);
    display.println("BTN: Start Recording");
    display.setCursor(0, 48);
    display.println("Hold: Drive + Record");
  } else if (mode == RECORDING || mode == DRIVING) {
    display.print(mode == DRIVING ? "DRIVING+REC " : "RECORDING ");
    display.println(filename);
    
    display.setCursor(0, 14);
    display.print("Time: ");
    display.print((millis() - recordStartTime) / 1000);
    display.println("s");
    
    display.setCursor(0, 26);
    display.print("Samples: ");
    display.println(sampleCount);
    
    display.setCursor(0, 38);
    display.print("F:");
    display.print(sF);
    display.print(" L:");
    display.print(sL);
    
    display.setCursor(0, 50);
    display.print("R:");
    display.print(sR);
    display.print(" E:");
    display.print(encCountL);
  }
  
  display.display();
}

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  delay(500);
  
  // Pin setup
  pinMode(BTN_PIN, INPUT_PULLUP);
  pinMode(STBY, OUTPUT);
  pinMode(AIN1, OUTPUT); pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT); pinMode(BIN2, OUTPUT);
  pinMode(PWMA, OUTPUT); pinMode(PWMB, OUTPUT);
  
  // Encoder pins
  pinMode(ENC_L_A, INPUT_PULLUP);
  pinMode(ENC_L_B, INPUT_PULLUP);
  pinMode(ENC_R_A, INPUT_PULLUP);
  pinMode(ENC_R_B, INPUT_PULLUP);
  
  // Encoder interrupts
  attachInterrupt(digitalPinToInterrupt(ENC_L_A), encLeftISR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_R_A), encRightISR, RISING);
  
  digitalWrite(STBY, HIGH);
  analogReadResolution(12);
  setForward();
  
  // Init buffers
  memset(fBuf, 0, sizeof(fBuf));
  memset(lBuf, 0, sizeof(lBuf));
  memset(rBuf, 0, sizeof(rBuf));
  
  // OLED
  display.begin(0x3C, true);
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);
  display.setTextSize(1);
  
  // SD Card
  if (!initSD()) {
    display.setCursor(0, 20);
    display.println("SD CARD ERROR!");
    display.println("Insert SD and reset");
    display.display();
    while(1);
  }
  
  Serial.println("=== Data Collector Ready ===");
  Serial.println("Short press: Record stationary");
  Serial.println("Long press: Drive + Record");
  
  updateDisplay();
}

// ==================== MAIN LOOP ====================
void loop() {
  static uint32_t btnPressTime = 0;
  static bool btnWasPressed = false;
  static uint32_t lastDisplayUpdate = 0;
  
  bool btnPressed = (digitalRead(BTN_PIN) == LOW);
  
  // Button handling
  if (btnPressed && !btnWasPressed) {
    btnPressTime = millis();
    btnWasPressed = true;
  }
  
  if (!btnPressed && btnWasPressed) {
    uint32_t pressDuration = millis() - btnPressTime;
    btnWasPressed = false;
    
    if (mode == IDLE) {
      if (pressDuration > 1000) {
        // Long press: Drive + Record
        mode = DRIVING;
        startNewFile();
      } else {
        // Short press: Record stationary (for calibration)
        mode = RECORDING;
        startNewFile();
      }
    } else {
      // Stop recording
      stopMotors();
      closeFile();
      mode = IDLE;
    }
  }
  
  // Recording logic
  if (mode == RECORDING || mode == DRIVING) {
    uint32_t now = millis();
    
    // Check if 1 minute elapsed
    if (now - recordStartTime >= RECORD_DURATION_MS) {
      closeFile();
      startNewFile();  // Start new file automatically
    }
    
    // Sample at fixed rate
    if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
      lastSampleTime = now;
      readSensors();
      
      // Driving mode: autonomous navigation
      if (mode == DRIVING) {
        curAction = 0;  // Forward
        
        // Emergency stop
        if (sF > FRONT_STOP) {
          stopMotors();
          curAction = 4;  // CRASH/STOP
          
          // Decide turn
          if (sL < SIDE_OPEN) {
            turnLeft90();
          } else if (sR < SIDE_OPEN) {
            turnRight90();
          } else {
            // 180
            turnLeft90();
            turnLeft90();
          }
          setForward();
        }
        // Left opening
        else if (sL < SIDE_OPEN && sF < FRONT_SLOW) {
          setMotors(BASE_SPD, BASE_SPD);
          delay(100);
          turnLeft90();
          setForward();
        }
        // Normal forward with PID
        else {
          int16_t spd = BASE_SPD;
          if (sF > FRONT_SLOW) {
            spd = map(sF, FRONT_SLOW, FRONT_STOP, MIN_SPD, BASE_SPD);
          }
          
          int16_t err = 0;
          if (sL > SIDE_WALL && sR > SIDE_WALL) {
            err = sL - sR;
          } else if (sL > SIDE_WALL) {
            err = sL - 2400;
          } else if (sR > SIDE_WALL) {
            err = 2400 - sR;
          }
          
          int16_t corr = (int16_t)(Kp * err + Kd * (err - lastErr));
          lastErr = err;
          corr = constrain(corr, -60, 60);
          
          int16_t pwmL = spd - corr;
          int16_t pwmR = spd + corr;
          
          // Crash avoidance
          if (sL > SIDE_CRASH) {
            pwmL += 40;
            pwmR -= 20;
            curAction = 4;
          }
          if (sR > SIDE_CRASH) {
            pwmL -= 20;
            pwmR += 40;
            curAction = 4;
          }
          
          setMotors(pwmL, pwmR);
        }
      }
      
      // Write sample
      writeSample();
    }
  }
  
  // Update display every 200ms
  if (millis() - lastDisplayUpdate > 200) {
    lastDisplayUpdate = millis();
    updateDisplay();
  }
}
