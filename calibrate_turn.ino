/*
 * MANUAL Encoder Calibration Tool
 * ================================
 * Shows LIVE encoder values on OLED display.
 * 
 * Instructions:
 * 1. Upload this code
 * 2. MANUALLY rotate each wheel by hand
 * 3. Watch encoder ticks on display
 * 4. Rotate robot body 90 degrees and note the ticks
 * 5. Press button to RESET counters to zero
 * 
 * This helps you find:
 * - Ticks per wheel revolution
 * - Ticks for 90-degree robot turn
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <Encoder.h>

// Pins
#define BTN_PIN 11

Encoder encL(1, 0);
Encoder encR(2, 8);
Adafruit_SH1106G display(128, 64, &Wire, -1);

void setup() {
  Serial.begin(115200);
  delay(100);
  
  Serial.println("\n========================================");
  Serial.println("    MANUAL ENCODER CALIBRATION");
  Serial.println("========================================");
  Serial.println("Rotate wheels BY HAND to see ticks");
  Serial.println("Press button to RESET to zero");
  Serial.println("========================================\n");
  
  pinMode(BTN_PIN, INPUT_PULLUP);
  
  // Init display
  display.begin(0x3C, true);
  display.setTextColor(SH110X_WHITE);
  display.clearDisplay();
  display.display();
  
  encL.write(0);
  encR.write(0);
}

void loop() {
  static uint32_t lastBtn = 0;
  static uint32_t lastUpdate = 0;
  
  // Button press = RESET encoders
  if (digitalRead(BTN_PIN) == LOW && millis() - lastBtn > 300) {
    lastBtn = millis();
    encL.write(0);
    encR.write(0);
    Serial.println(">>> RESET TO ZERO <<<");
  }
  
  // Update display every 50ms
  if (millis() - lastUpdate > 50) {
    lastUpdate = millis();
    
    long posL = encL.read();
    long posR = encR.read();
    
    // OLED Display
    display.clearDisplay();
    
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("ENCODER CALIBRATION");
    display.println("Rotate wheels by hand");
    display.println("BTN = Reset to 0");
    
    display.drawLine(0, 26, 128, 26, SH110X_WHITE);
    
    display.setTextSize(2);
    display.setCursor(0, 32);
    display.print("L:");
    display.println(posL);
    
    display.setCursor(0, 50);
    display.print("R:");
    display.println(posR);
    
    display.display();
    
    // Serial output
    Serial.print("L: ");
    Serial.print(posL);
    Serial.print("\tR: ");
    Serial.print(posR);
    Serial.print("\tDiff: ");
    Serial.println(posL - posR);
  }
}
