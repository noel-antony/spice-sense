/*
 * Spice Purity Inference - Raw Sensor Model
 * 
 * This version uses only 3 direct sensor readings:
 * 1. MQ135 AQI (calculated from Rs/R0 ratio)
 * 2. MQ3 Voltage
 * 3. BME688 Gas Resistance (kOhm)
 * 
 * Hardware:
 * - ESP32 DevKit
 * - BME688 on I2C (SDA=21, SCL=22)
 * - MQ135 on GPIO34 (ADC)
 * - MQ3 on GPIO35 (ADC)
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

// ============================================================
// SENSOR CALIBRATION
// ============================================================
#define SEALEVELPRESSURE_HPA 1013.25

// MQ135 calibration (Rs/R0 in clean air ~3.6)
#define MQ135_R0 10.0       // Calibrated R0 in kOhm
#define MQ135_RL 1.0        // Load resistance in kOhm
#define MQ135_CLEAN_AIR_RATIO 3.6

// MQ3 calibration
#define MQ3_R0 10.0
#define MQ3_RL 1.0

// ============================================================
// RAW SENSOR MODEL PARAMETERS (from Python training)
// ============================================================

// Features: [mq135_aqi, mq3_voltage, gas_resistance_kOhm]
const int NUM_FEATURES = 3;

// Feature scaling parameters (StandardScaler)
const float FEATURE_MEAN[3] = {61.7027, 1.2819, 34.8238};
const float FEATURE_STD[3] = {13.7440, 0.0363, 3.7955};

// Gas resistance thresholds for multi-class classification
// Note: LOWER gas resistance = HIGHER purity (cleaner spice)
const float GAS_BEST_GOOD = 29.85;   // < this = BEST (100% pure)
const float GAS_GOOD_POOR = 34.11;   // < this = GOOD (75%+)
const float GAS_POOR_WORST = 37.23;  // < this = POOR (50%+), else WORST

// Binary classification threshold
const float GAS_HIGH_LOW = 33.86;    // < this = high purity (>=75%)

// Purity lookup table for interpolation
const int NUM_LOOKUP_POINTS = 4;
const float GAS_POINTS[4] = {26.90, 32.81, 35.41, 39.06};
const float PURITY_POINTS[4] = {100.0, 75.0, 50.0, 25.0};

// Linear regression parameters
// purity = INTERCEPT + COEF * gas_resistance
const float REG_INTERCEPT = 257.1628;
const float REG_COEF_GAS = -5.7965;

// Multi-feature regression (backup)
// purity = INTERCEPT + c[0]*aqi + c[1]*mq3_v + c[2]*gas_r
const float REG_MULTI_INTERCEPT = 231.2104;
const float REG_MULTI_COEF[3] = {-0.0847, 31.4481, -6.0510};

// ============================================================
// STABILIZATION DETECTION
// ============================================================
#define WARMUP_SECONDS 60
#define STABLE_READINGS 10
#define VARIANCE_THRESHOLD 5.0

// ============================================================
// GLOBAL VARIABLES
// ============================================================
Adafruit_BME680 bme;

// Sensor readings
float gasResistance_kOhm = 0;
float mq135_aqi = 0;
float mq3_voltage = 0;
float mq135_ratio = 0;

// Stabilization tracking
float gasHistory[STABLE_READINGS];
int historyIndex = 0;
bool isStabilized = false;
unsigned long startTime = 0;

// ============================================================
// SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    
    Serial.println("\n========================================");
    Serial.println("SPICE PURITY TESTER - RAW SENSOR MODEL");
    Serial.println("========================================");
    Serial.println("Features: MQ135 AQI, MQ3 Voltage, Gas Resistance");
    
    // Initialize I2C
    Wire.begin(I2C_SDA, I2C_SCL);
    
    // Initialize BME688
    if (!bme.begin()) {
        Serial.println("ERROR: BME688 not found!");
        while (1) delay(100);
    }
    Serial.println("✓ BME688 initialized");
    
    // Configure BME688
    bme.setTemperatureOversampling(BME680_OS_8X);
    bme.setHumidityOversampling(BME680_OS_2X);
    bme.setPressureOversampling(BME680_OS_4X);
    bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
    bme.setGasHeater(320, 150);  // 320°C for 150ms
    
    // Initialize ADC pins
    analogReadResolution(12);  // 12-bit ADC (0-4095)
    pinMode(MQ135_PIN, INPUT);
    pinMode(MQ3_PIN, INPUT);
    Serial.println("✓ MQ sensors initialized");
    
    // Initialize history
    for (int i = 0; i < STABLE_READINGS; i++) {
        gasHistory[i] = 0;
    }
    
    startTime = millis();
    
    Serial.println("\n⏳ Warming up sensors (60 seconds)...");
    Serial.println("========================================\n");
}

// ============================================================
// SENSOR READING FUNCTIONS
// ============================================================

float readMQ135_AQI() {
    int raw = analogRead(MQ135_PIN);
    float voltage = raw * (3.3 / 4095.0);
    
    // Calculate sensor resistance
    float rs = MQ135_RL * (3.3 - voltage) / voltage;
    
    // Calculate ratio Rs/R0
    mq135_ratio = rs / MQ135_R0;
    
    // Calculate AQI (inverse relationship)
    // Higher ratio = cleaner air = lower AQI
    if (mq135_ratio <= 0) return 500.0;
    float aqi = 100.0 / mq135_ratio;
    
    // Clamp to valid range
    if (aqi < 0) aqi = 0;
    if (aqi > 500) aqi = 500;
    
    return aqi;
}

float readMQ3_Voltage() {
    int raw = analogRead(MQ3_PIN);
    return raw * (3.3 / 4095.0);
}

float readBME688_GasResistance() {
    if (bme.performReading()) {
        return bme.gas_resistance / 1000.0;  // Convert to kOhm
    }
    return -1;
}

// ============================================================
// STABILIZATION CHECK
// ============================================================

bool checkStabilization(float gasR) {
    // Add to history
    gasHistory[historyIndex] = gasR;
    historyIndex = (historyIndex + 1) % STABLE_READINGS;
    
    // Calculate variance
    float sum = 0, sumSq = 0;
    for (int i = 0; i < STABLE_READINGS; i++) {
        sum += gasHistory[i];
        sumSq += gasHistory[i] * gasHistory[i];
    }
    float mean = sum / STABLE_READINGS;
    float variance = (sumSq / STABLE_READINGS) - (mean * mean);
    
    // Check if warmup complete and readings stable
    unsigned long elapsed = (millis() - startTime) / 1000;
    return (elapsed >= WARMUP_SECONDS) && (variance < VARIANCE_THRESHOLD);
}

// ============================================================
// ML INFERENCE FUNCTIONS
// ============================================================

float scaleFeature(float value, int featureIndex) {
    return (value - FEATURE_MEAN[featureIndex]) / FEATURE_STD[featureIndex];
}

// Binary classification: High purity (>=75%) vs Low purity
String predictBinary(float gasR) {
    return (gasR < GAS_HIGH_LOW) ? "HIGH_PURITY" : "LOW_PURITY";
}

// Multi-class classification: BEST/GOOD/POOR/WORST
String predictMultiClass(float gasR) {
    if (gasR < GAS_BEST_GOOD) return "BEST";
    if (gasR < GAS_GOOD_POOR) return "GOOD";
    if (gasR < GAS_POOR_WORST) return "POOR";
    return "WORST";
}

// Regression: Exact purity percentage (using interpolation)
float predictPurityInterpolation(float gasR) {
    // Handle edge cases
    if (gasR <= GAS_POINTS[0]) return PURITY_POINTS[0];  // 100%
    if (gasR >= GAS_POINTS[NUM_LOOKUP_POINTS - 1]) return PURITY_POINTS[NUM_LOOKUP_POINTS - 1];  // 25%
    
    // Linear interpolation
    for (int i = 0; i < NUM_LOOKUP_POINTS - 1; i++) {
        if (gasR >= GAS_POINTS[i] && gasR < GAS_POINTS[i + 1]) {
            float ratio = (gasR - GAS_POINTS[i]) / (GAS_POINTS[i + 1] - GAS_POINTS[i]);
            return PURITY_POINTS[i] + ratio * (PURITY_POINTS[i + 1] - PURITY_POINTS[i]);
        }
    }
    return 50.0;  // Default middle value
}

// Regression: Using linear equation
float predictPurityLinear(float gasR) {
    float purity = REG_INTERCEPT + REG_COEF_GAS * gasR;
    // Clamp to valid range
    if (purity < 0) purity = 0;
    if (purity > 100) purity = 100;
    return purity;
}

// Regression: Using multi-feature equation
float predictPurityMultiFeature(float aqi, float mq3_v, float gasR) {
    float purity = REG_MULTI_INTERCEPT + 
                   REG_MULTI_COEF[0] * aqi + 
                   REG_MULTI_COEF[1] * mq3_v + 
                   REG_MULTI_COEF[2] * gasR;
    // Clamp to valid range
    if (purity < 0) purity = 0;
    if (purity > 100) purity = 100;
    return purity;
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
    // Read sensors
    gasResistance_kOhm = readBME688_GasResistance();
    mq135_aqi = readMQ135_AQI();
    mq3_voltage = readMQ3_Voltage();
    
    // Check stabilization
    bool wasStabilized = isStabilized;
    isStabilized = checkStabilization(gasResistance_kOhm);
    
    // Print status
    unsigned long elapsed = (millis() - startTime) / 1000;
    
    Serial.println("----------------------------------------");
    Serial.printf("Time: %lu seconds\n", elapsed);
    
    // Print raw sensor values
    Serial.println("\n📊 RAW SENSOR VALUES:");
    Serial.printf("   MQ135 AQI:        %.2f\n", mq135_aqi);
    Serial.printf("   MQ135 Ratio:      %.2f\n", mq135_ratio);
    Serial.printf("   MQ3 Voltage:      %.3f V\n", mq3_voltage);
    Serial.printf("   Gas Resistance:   %.2f kOhm\n", gasResistance_kOhm);
    
    // Print scaled features
    Serial.println("\n📐 SCALED FEATURES:");
    Serial.printf("   AQI (scaled):     %.4f\n", scaleFeature(mq135_aqi, 0));
    Serial.printf("   MQ3 V (scaled):   %.4f\n", scaleFeature(mq3_voltage, 1));
    Serial.printf("   Gas R (scaled):   %.4f\n", scaleFeature(gasResistance_kOhm, 2));
    
    // Stabilization status
    if (!isStabilized) {
        Serial.println("\n⏳ WARMING UP...");
        Serial.printf("   Time remaining: ~%d seconds\n", 
                      max(0, (int)(WARMUP_SECONDS - elapsed)));
    } else {
        if (!wasStabilized) {
            Serial.println("\n✅ SENSORS STABILIZED!");
        }
        
        // Run predictions
        Serial.println("\n🔮 PREDICTIONS:");
        
        // Binary
        String binaryResult = predictBinary(gasResistance_kOhm);
        Serial.printf("   Binary:      %s\n", binaryResult.c_str());
        
        // Multi-class
        String multiResult = predictMultiClass(gasResistance_kOhm);
        Serial.printf("   Category:    %s\n", multiResult.c_str());
        
        // Regression (multiple methods)
        float purityInterp = predictPurityInterpolation(gasResistance_kOhm);
        float purityLinear = predictPurityLinear(gasResistance_kOhm);
        float purityMulti = predictPurityMultiFeature(mq135_aqi, mq3_voltage, gasResistance_kOhm);
        
        Serial.println("\n📈 PURITY ESTIMATES:");
        Serial.printf("   Interpolation: %.1f%%\n", purityInterp);
        Serial.printf("   Linear Reg:    %.1f%%\n", purityLinear);
        Serial.printf("   Multi-Feature: %.1f%%\n", purityMulti);
        
        // Final recommendation (use interpolation as primary)
        Serial.println("\n" + String("═").repeat(40));
        Serial.printf("🎯 FINAL: %.0f%% PURE - %s\n", purityInterp, multiResult.c_str());
        Serial.println(String("═").repeat(40));
    }
    
    Serial.println();
    delay(2000);  // 2 second delay between readings
}

// Helper function for string repeat (not built into Arduino String)
String repeatChar(char c, int n) {
    String result = "";
    for (int i = 0; i < n; i++) result += c;
    return result;
}
