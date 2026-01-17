/*
 * Spice Purity ESP32 - Serial Output
 * Outputs sensor data via Serial for PC server processing
 * 
 * Hardware:
 * - ESP32 DevKit
 * - BME688 on I2C (SDA=21, SCL=22)
 * - MQ135 on GPIO34 (ADC)
 * - MQ3 on GPIO35 (ADC)
 * 
 * Connect USB to PC and run serial_bridge.py
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include "Adafruit_BME680.h"

// ============================================================
// PIN DEFINITIONS
// ============================================================
#define MQ135_PIN 34
#define MQ3_PIN 35
#define I2C_SDA 21
#define I2C_SCL 22
#define LED_PIN 2  // Built-in LED

// ============================================================
// SENSOR CALIBRATION
// ============================================================
#define MQ135_R0 10.0       // Calibrated R0 in kOhm
#define MQ135_RL 1.0        // Load resistance in kOhm
#define MQ3_R0 10.0
#define MQ3_RL 1.0

// ============================================================
// TIMING
// ============================================================
#define WARMUP_SECONDS 60
#define READ_INTERVAL_MS 2000  // Read every 2 seconds

// ============================================================
// GLOBAL VARIABLES
// ============================================================
Adafruit_BME680 bme;
unsigned long startTime = 0;
unsigned long lastReadTime = 0;
bool isWarmedUp = false;
int readingCount = 0;

// ============================================================
// SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    
    Serial.println();
    Serial.println("========================================");
    Serial.println("SPICE PURITY TESTER - Serial Mode");
    Serial.println("========================================");
    Serial.println("Connect this device to PC and run:");
    Serial.println("  python serial_bridge.py");
    Serial.println("========================================");
    
    // Initialize I2C
    Wire.begin(I2C_SDA, I2C_SCL);
    
    // Initialize BME688
    if (!bme.begin()) {
        Serial.println("ERROR: BME688 not found!");
        while (1) {
            digitalWrite(LED_PIN, !digitalRead(LED_PIN));
            delay(100);
        }
    }
    Serial.println("BME688: OK");
    
    // Configure BME688
    bme.setTemperatureOversampling(BME680_OS_8X);
    bme.setHumidityOversampling(BME680_OS_2X);
    bme.setPressureOversampling(BME680_OS_4X);
    bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
    bme.setGasHeater(320, 150);
    
    // Initialize ADC
    analogReadResolution(12);
    pinMode(MQ135_PIN, INPUT);
    pinMode(MQ3_PIN, INPUT);
    Serial.println("MQ Sensors: OK");
    
    startTime = millis();
    Serial.println();
    Serial.println("Warming up sensors (60 seconds)...");
    Serial.println("========================================");
}

// ============================================================
// SENSOR READING FUNCTIONS
// ============================================================
float readMQ135_Ratio() {
    int raw = analogRead(MQ135_PIN);
    float voltage = raw * (3.3 / 4095.0);
    
    if (voltage <= 0.01) return 999.0;
    
    float rs = MQ135_RL * (3.3 - voltage) / voltage;
    return rs / MQ135_R0;
}

float readMQ3_Voltage() {
    int raw = analogRead(MQ3_PIN);
    return raw * (3.3 / 4095.0);
}

float readGasResistance_kOhm() {
    if (bme.performReading()) {
        return bme.gas_resistance / 1000.0;
    }
    return -1;
}

// ============================================================
// MAIN LOOP
// ============================================================
void loop() {
    unsigned long now = millis();
    unsigned long elapsed = (now - startTime) / 1000;
    
    // Check warmup status
    if (!isWarmedUp && elapsed >= WARMUP_SECONDS) {
        isWarmedUp = true;
        Serial.println();
        Serial.println("========================================");
        Serial.println("SENSORS READY - Starting data output");
        Serial.println("========================================");
        Serial.println();
        digitalWrite(LED_PIN, HIGH);
    }
    
    // Read and output at interval
    if (now - lastReadTime >= READ_INTERVAL_MS) {
        lastReadTime = now;
        
        // Read sensors
        float mq135_ratio = readMQ135_Ratio();
        float mq3_voltage = readMQ3_Voltage();
        float gas_resistance = readGasResistance_kOhm();
        
        if (!isWarmedUp) {
            // During warmup, show countdown
            int remaining = WARMUP_SECONDS - elapsed;
            Serial.print("Warmup: ");
            Serial.print(remaining);
            Serial.print("s | Gas: ");
            Serial.print(gas_resistance, 1);
            Serial.println(" kOhm");
        } else {
            // Output in parseable format
            // Format: "MQ135 Ratio: X.XX | MQ3: X.XXV | Gas: XX.XX kΩ"
            readingCount++;
            
            Serial.print("[");
            Serial.print(readingCount);
            Serial.print("] MQ135 Ratio: ");
            Serial.print(mq135_ratio, 2);
            Serial.print(" | MQ3: ");
            Serial.print(mq3_voltage, 3);
            Serial.print("V | Gas: ");
            Serial.print(gas_resistance, 2);
            Serial.println(" kOhm");
            
            // Blink LED to indicate data sent
            digitalWrite(LED_PIN, LOW);
            delay(50);
            digitalWrite(LED_PIN, HIGH);
        }
    }
}
