#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Encoder.h>
#include <SD.h>
#include "weights.h" 

// --- 1. CONFIGURATION (Tune these for your maze) ---
const long TICKS_90 = 455;   // Encoder counts for a perfect 90-degree turn
const int MAX_SPD    = 210;   // Speed in clear straights
const int TURN_SPD   = 130;   // Speed during pivot turns
const int MIN_PWM    = 65;    // Minimum power to move

// Safety Thresholds (ADC values)
#define FRONT_CRITICAL 2800   // Hard stop and turn
#define FRONT_SLOW     1800   // Start deceleration
#define SIDE_DANGER    3400   // Immediate nudge away
#define CORRIDOR_AVG   2400   // Target for centering

// --- 2. HARDWARE PINS ---
#define PWMA 9
#define PWMB 10
#define AIN1 4
#define AIN2 3
#define BIN1 6
#define BIN2 7
#define STBY 5
#define BTN  11
const uint8_t SH_F = 21, SH_L = 23, SH_R = 22;

Encoder encL(1, 0); 
Encoder encR(2, 8);
Adafruit_SH1106G display(128, 64, &Wire, -1);

// --- 3. STATE VARIABLES ---
enum State { IDLE, AUTO_RUN };
State botState = IDLE;
int lastError = 0;

void setForward() {
  digitalWrite(AIN1, HIGH); digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, HIGH); digitalWrite(BIN2, LOW);
}

void drive(int l, int r) {
  analogWrite(PWMA, l);
  analogWrite(PWMB, r);
}

// Precision Encoder Turn
void executeTurn(long tl, long tr) {
  encL.write(0); encR.write(0);
  digitalWrite(AIN1, tl > 0 ? HIGH : LOW); digitalWrite(AIN2, tl > 0 ? LOW : HIGH);
  digitalWrite(BIN1, tr > 0 ? HIGH : LOW); digitalWrite(BIN2, tr > 0 ? LOW : HIGH);
  
  while (abs(encL.read()) < abs(tl) || abs(encR.read()) < abs(tr)) {
    analogWrite(PWMA, abs(encL.read()) < abs(tl) ? TURN_SPD : 0);
    analogWrite(PWMB, abs(encR.read()) < abs(tr) ? TURN_SPD : 0);
  }
  drive(0, 0); delay(100); setForward();
}

void setup() {
  pinMode(PWMA, OUTPUT); pinMode(PWMB, OUTPUT);
  pinMode(AIN1, OUTPUT); pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT); pinMode(BIN2, OUTPUT);
  pinMode(STBY, OUTPUT); pinMode(BTN, INPUT_PULLUP);
  
  digitalWrite(STBY, HIGH);
  analogReadResolution(12);
  display.begin(0x3C, true);
  display.setTextColor(SH110X_WHITE);
  setForward();
}

void loop() {
  if (digitalRead(BTN) == LOW) {
    botState = (botState == IDLE) ? AUTO_RUN : IDLE;
    if (botState == IDLE) drive(0, 0);
    delay(500);
  }

  if (botState == AUTO_RUN) {
    int f = analogRead(SH_F), l = analogRead(SH_L), r = analogRead(SH_R);

    // 1. NEURAL NETWORK INFERENCE (32x16 Architecture)
    float in[] = {(float)f/4095.0f, (float)l/4095.0f, (float)r/4095.0f};
    float h1[32], h2[16];
    for(int j=0; j<32; j++) {
      float s = b1[j]; for(int i=0; i<3; i++) s += in[i]*w1[j][i];
      h1[j] = tanhf(s);
    }
    for(int j=0; j<16; j++) {
      float s = b2[j]; for(int i=0; i<32; i++) s += h1[i]*w2[j][i];
      h2[j] = tanhf(s);
    }
    float outL = b3[0], outR = b3[1];
    for(int i=0; i<16; i++) { outL += h2[i]*w3[0][i]; outR += h2[i]*w3[1][i]; }
    int aiBias = (outL - outR) * 25; // Controlled influence

    // 2. PD CENTERING (Predictive Stability)
    int error = l - r;
    int dErr = error - lastError;
    lastError = error;
    int correction = (int)(0.04 * error + 0.02 * dErr);

    // 3. PREDICTIVE BRAKING & SPEED SCALING
    int speed = (f > FRONT_SLOW) ? MIN_PWM : map(f, 0, FRONT_SLOW, MAX_SPD, MIN_PWM);
    
    // 4. COMBINE & CONSTRAIN
    int sL = speed - correction - aiBias;
    int sR = speed + correction + aiBias;

    // 5. HARD SAFETY OVERRIDES
    if (f > FRONT_CRITICAL) {
      drive(0, 0); delay(100);
      if (l < r) executeTurn(-TICKS_90, TICKS_90); // Turn Left
      else executeTurn(TICKS_90, -TICKS_90);        // Turn Right
      return;
    }

    if (l > SIDE_DANGER) sL += 70; // Hard nudge Right
    if (r > SIDE_DANGER) sR += 70; // Hard nudge Left

    drive(constrain(sL, MIN_PWM, 250), constrain(sR, MIN_PWM, 250));
    delay(20); 
  }

  // Telemetry
  display.clearDisplay();
  display.setCursor(0, 0);
  display.printf("F:%d\nL:%d R:%d", analogRead(SH_F), analogRead(SH_L), analogRead(SH_R));
  display.display();
}