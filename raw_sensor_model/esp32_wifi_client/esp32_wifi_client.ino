/*
 * Spice Purity ESP32 - WiFi Client
 * Sends sensor data to PC server for ML inference
 * 
 * Hardware:
 * - ESP32 DevKit
 * - BME688 on I2C (SDA=21, SCL=22)
 * - MQ135 on GPIO34 (ADC)
 * - MQ3 on GPIO35 (ADC)
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include "Adafruit_BME680.h"

// ============================================================
// WIFI CONFIGURATION - UPDATE THESE!
// ============================================================
const char* WIFI_SSID = "YOUR_WIFI_SSID";      // Your WiFi network name
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";  // Your WiFi password
const char* SERVER_URL = "https://spice-purity-server.onrender.com/api/predict";  // Cloud server URL

// ============================================================
// PIN DEFINITIONS
// ============================================================
#define MQ135_PIN 34
#define MQ3_PIN 35
#define I2C_SDA 21
#define I2C_SCL 22
#define LED_PIN 2  // Built-in LED for status

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
#define SEND_INTERVAL_MS 2000  // Send data every 2 seconds

// ============================================================
// GLOBAL VARIABLES
// ============================================================
Adafruit_BME680 bme;
unsigned long startTime = 0;
unsigned long lastSendTime = 0;
bool isWarmedUp = false;

// ============================================================
// SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    
    Serial.println("\n========================================");
    Serial.println("SPICE PURITY TESTER - WiFi Client");
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
    Serial.println("✓ BME688 initialized");
    
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
    Serial.println("✓ MQ sensors initialized");
    
    // Connect to WiFi
    connectWiFi();
    
    startTime = millis();
    Serial.println("\n⏳ Warming up sensors (60 seconds)...");
}

// ============================================================
// WIFI CONNECTION
// ============================================================
void connectWiFi() {
    Serial.printf("\nConnecting to WiFi: %s", WIFI_SSID);
    
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n✓ WiFi connected!");
        Serial.printf("  IP Address: %s\n", WiFi.localIP().toString().c_str());
        digitalWrite(LED_PIN, HIGH);
    } else {
        Serial.println("\n✗ WiFi connection failed!");
        digitalWrite(LED_PIN, LOW);
    }
}

// ============================================================
// SENSOR READING FUNCTIONS
// ============================================================
float readMQ135_Ratio() {
    int raw = analogRead(MQ135_PIN);
    float voltage = raw * (3.3 / 4095.0);
    
    if (voltage <= 0.01) return 999.0;  // Prevent division by zero
    
    // Calculate sensor resistance
    float rs = MQ135_RL * (3.3 - voltage) / voltage;
    
    // Return Rs/R0 ratio
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
// SEND DATA TO SERVER
// ============================================================
void sendDataToServer(float mq135_ratio, float mq3_voltage, float gas_resistance) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("⚠️ WiFi disconnected, reconnecting...");
        connectWiFi();
        return;
    }
    
    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    
    // Create JSON payload
    StaticJsonDocument<200> doc;
    doc["mq135_ratio"] = mq135_ratio;
    doc["mq3_voltage"] = mq3_voltage;
    doc["gas_resistance_kOhm"] = gas_resistance;
    
    String jsonPayload;
    serializeJson(doc, jsonPayload);
    
    Serial.printf("📤 Sending: %s\n", jsonPayload.c_str());
    
    int httpCode = http.POST(jsonPayload);
    
    if (httpCode > 0) {
        if (httpCode == HTTP_CODE_OK) {
            String response = http.getString();
            
            // Parse response
            StaticJsonDocument<512> responseDoc;
            DeserializationError error = deserializeJson(responseDoc, response);
            
            if (!error) {
                float purity = responseDoc["predictions"]["regression"]["purity_percent"];
                const char* quality = responseDoc["predictions"]["multiclass"]["label"];
                
                Serial.println("----------------------------------------");
                Serial.printf("🎯 PURITY: %.1f%% - %s\n", purity, quality);
                Serial.println("----------------------------------------");
                
                // Blink LED based on quality
                blinkQuality(quality);
            }
        } else {
            Serial.printf("⚠️ HTTP Error: %d\n", httpCode);
        }
    } else {
        Serial.printf("❌ Connection failed: %s\n", http.errorToString(httpCode).c_str());
    }
    
    http.end();
}

void blinkQuality(const char* quality) {
    // Quick blink pattern based on quality
    int blinks = 1;
    if (strcmp(quality, "BEST") == 0) blinks = 4;
    else if (strcmp(quality, "GOOD") == 0) blinks = 3;
    else if (strcmp(quality, "POOR") == 0) blinks = 2;
    else blinks = 1;
    
    for (int i = 0; i < blinks; i++) {
        digitalWrite(LED_PIN, LOW);
        delay(100);
        digitalWrite(LED_PIN, HIGH);
        delay(100);
    }
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
        Serial.println("\n✅ Sensors warmed up! Starting data transmission...\n");
    }
    
    // Read sensors
    float mq135_ratio = readMQ135_Ratio();
    float mq3_voltage = readMQ3_Voltage();
    float gas_resistance = readGasResistance_kOhm();
    
    // Print local readings
    Serial.printf("\n📊 MQ135 Ratio: %.2f | MQ3: %.3fV | Gas: %.2f kΩ\n",
                  mq135_ratio, mq3_voltage, gas_resistance);
    
    if (!isWarmedUp) {
        Serial.printf("⏳ Warming up... %d/%d seconds\n", (int)elapsed, WARMUP_SECONDS);
    } else {
        // Send data at interval
        if (now - lastSendTime >= SEND_INTERVAL_MS) {
            sendDataToServer(mq135_ratio, mq3_voltage, gas_resistance);
            lastSendTime = now;
        }
    }
    
    delay(500);  // Small delay between readings
}
