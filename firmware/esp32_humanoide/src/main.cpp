/*
 * Humanoide 12 DOF: ESP32 + PCA9685 + MPU-6050 <-> Host Python
 * ------------------------------------------------------------
 * Evolucao do firmware de bancada (esp32_pca9685_servo_test v1.1) com:
 *   - mapa canal <-> junta, limites, zero, sentido e velocidade maxima por canal
 *   - modo de configuracao (CFG) separado do modo de operacao (RUN)
 *   - configuracao gravada na flash (NVS), carregada no boot
 *   - comandos em coordenadas de junta (J, JALL) usados pelo controlador
 *
 * Hardware:
 *   - ESP32 DevKit V1 (ESP32-WROOM-32)
 *   - PCA9685 (0x40) com V+ = 5..6 V no borne | MPU-6050 (0x68/0x69) no mesmo I2C
 *   - GPIO21->SDA | GPIO22->SCL | GPIO23->OE (opcional, ativo em LOW)
 *
 * Build/upload (nesta pasta):  pio run --target upload  |  pio device monitor
 *
 * Transportes: USB Serial (115200 8N1) e Bluetooth Classic SPP (BT_NAME / BT_PIN),
 * mesmo protocolo; cada resposta volta pela porta de onde veio o comando e o
 * heartbeat (STATUS) vai so para a USB.
 *
 * Protocolo (ASCII, '\n'; maiusculas/minusculas tanto faz):
 *  Operacao (qualquer modo)
 *   POS:<canal>:<angulo>:<ms>     angulo do servo 0-180. RUN: rejeita fora de [min,max]
 *   J:<junta>:<graus>:<ms>        graus da junta (0 = zero calibrado), satura nos limites
 *   JALL:<ms>:<g0>:...:<g11>      todas as juntas na ordem de JOINTS; "X" = nao mexe
 *   CENTER[:<canal>]              vai para 90 (saturado nos limites)
 *   REST                          todas as juntas para o zero calibrado
 *   STATUS | IMU | SCAN | ON | OFF | PING | INFO | HELP
 *   IMUSTREAM:<ms>                amostras I:<t>:<ax>:<ay>:<az>:<gx>:<gy>:<gz> a cada ms (0 = para), pela porta que pediu
 *   JOINTS                        lista as juntas na ordem do JALL
 *   CFG                           configuracao completa em JSON
 *  Modo
 *   MODE:CFG | MODE:RUN           CFG libra os comandos de configuracao e permite
 *                                 POS fora dos limites (para achar o curso mecanico),
 *                                 com velocidade limitada a CFG_VEL_DPS. J/JALL bloqueados.
 *  Configuracao (so em MODE:CFG; vale na hora, SAVE grava na flash)
 *   MAP:<canal>:<junta|->         associa a junta ao canal (- desassocia)
 *   LIM:<canal>:<min>:<max>       limites do angulo do servo
 *   SETMIN:<canal> | SETMAX:<canal> | SETZERO:<canal>   usa o angulo atual do canal
 *   ZERO:<canal>:<angulo>         angulo do servo que corresponde a junta em 0 grau
 *   DIR:<canal>:<1|-1>            sentido: graus da junta -> graus do servo
 *   VEL:<canal>:<graus/s>         velocidade maxima (0 = sem limite)
 *   EN:<canal>:<0|1>              habilita/desliga o canal (desligado = sem pulso)
 *   WIGGLE:<canal>                balanca +-8 graus para identificar a junta
 *   SAVE | LOAD | DEFAULTS        grava / recarrega da flash / volta ao padrao (sem gravar)
 *
 * Respostas: ACK | PONG | JSON | ERR:INVALID_CMD | ERR:OUT_OF_BOUNDS | ERR:LIMIT:<min>:<max>
 *            ERR:NOT_MAPPED | ERR:DISABLED | ERR:NEED_CFG_MODE | ERR:IN_CFG_MODE | ERR:UNKNOWN_JOINT
 */

#include <Arduino.h>
#include <Wire.h>
#include <Preferences.h>
#include <Adafruit_PWMServoDriver.h>

#define USE_BT        true        // Bluetooth Classic (SPP): mesmo protocolo da USB
#if USE_BT
#include "BluetoothSerial.h"
#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error "Bluetooth Classic nao habilitado neste SDK (use ESP32 original, nao S3/C3)"
#endif
#endif

// ---------- Configuracao de hardware ----------
#define I2C_SDA_PIN   21
#define I2C_SCL_PIN   22
#define PCA_I2C_ADDR  0x40
#define OE_PIN        23
#define USE_OE        true        // false se o pino OE nao estiver ligado
#define SERVO_FREQ    50          // 50 Hz -> periodo de 20 ms
#define SERVOMIN      111         // ~544 us  (ajustar por lote de servo)
#define SERVOMAX      491         // ~2400 us (ajustar por lote de servo)
#define NUM_CHANNELS  16
#define CENTER_ON_BOOT true       // vai para o zero calibrado de cada canal ao ligar

// ---------- MPU-6050 (IMU) ----------
#define MPU_ADDR_PRIMARY   0x68
#define MPU_ADDR_SECONDARY 0x69
#define MPU_REG_CONFIG     0x1A
#define MPU_REG_GYRO_CFG   0x1B
#define MPU_REG_ACCEL_CFG  0x1C
#define MPU_REG_ACCEL_XOUT 0x3B
#define MPU_REG_PWR_MGMT_1 0x6B

// ---------- Comunicacao / temporizacao ----------
#define BAUD_RATE       115200
#define SERVO_UPDATE_MS 20        // interpolacao a 50 Hz
#define STREAM_MS       40        // duracao <= isso (sem limite de vel.) = quadro de fluxo, aplica direto
#define TELEMETRY_MS    1000      // heartbeat STATUS (so USB)
#define CMD_LINE_MAX    200       // JALL com 12 juntas cabe com folga
#define MAX_TOKENS      16
#define FW_VERSION      "2.2"
#define CFG_VEL_DPS     60.0f     // velocidade maxima em MODE:CFG
#define WIGGLE_DEG      8.0f

// ---------- Bluetooth (SPP) ----------
#define BT_NAME "Humanoide-Bancada"   // nome que aparece ao parear no PC/celular
#define BT_PIN  "1234"                // PIN de pareamento (4 digitos) - TROQUE antes de uso real

// ---------- Juntas (mesma ordem do modelo/robo.json) ----------
const char *JOINT_NAMES[] = {
  "quadril_roll_L", "quadril_pitch_L", "joelho_L", "tornozelo_L",
  "quadril_roll_R", "quadril_pitch_R", "joelho_R", "tornozelo_R",
  "ombro_L", "cotovelo_L", "ombro_R", "cotovelo_R",
};
const int8_t NUM_JOINTS = sizeof(JOINT_NAMES) / sizeof(JOINT_NAMES[0]);

// ---------- Configuracao por canal (gravada na NVS) ----------
#define CFG_MAGIC 0x48554D32      // "HUM2": muda se o struct mudar
struct ServoCfg {
  int8_t joint;      // indice em JOINT_NAMES, -1 = sem junta
  int8_t dir;        // +1 / -1
  uint8_t enabled;
  float minDeg;      // limites do angulo do servo
  float maxDeg;
  float zeroDeg;     // angulo do servo com a junta em 0 grau
  float maxVel;      // graus/s, 0 = sem limite
};
struct ConfigStore {
  uint32_t magic;
  ServoCfg servo[NUM_CHANNELS];
};
ConfigStore cfg;
Preferences prefs;
bool cfgMode = false;
bool cfgDirty = false;    // ha alteracoes ainda nao gravadas

Adafruit_PWMServoDriver pwm(PCA_I2C_ADDR);

#if USE_BT
BluetoothSerial SerialBT;
#endif
// Destino das respostas: a porta (USB ou Bluetooth) de onde veio o comando atual.
Print *out = &Serial;

struct ChannelMove {
  float startAngle = 90.0f;
  float currentAngle = 90.0f;
  float targetAngle = 90.0f;
  unsigned long t0 = 0;
  unsigned long duration = 0;
  bool moving = false;
  bool eased = true;      // false = linear (re-alvo no meio de um movimento)
  bool clamped = false;   // ultimo comando de junta foi saturado no limite
  uint8_t wiggle = 0;     // fases restantes do WIGGLE
  float wiggleCenter = 90.0f;
};

ChannelMove channels[NUM_CHANNELS];
String rxLine;      // buffer de linha da USB
String rxLineBT;    // buffer de linha do Bluetooth
unsigned long lastServoUpdate = 0;
unsigned long lastTelemetry = 0;
unsigned long imuStreamMs = 0;     // IMUSTREAM: periodo das amostras (0 = desligado)
unsigned long lastImuStream = 0;
Print *imuOut = nullptr;           // porta que pediu o fluxo (USB ou Bluetooth)

// ---------- Configuracao ----------
void defaultConfig() {
  cfg.magic = CFG_MAGIC;
  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    cfg.servo[i] = {-1, 1, 1, 0.0f, 180.0f, 90.0f, 0.0f};
  }
}

bool loadConfig() {
  prefs.begin("humanoide", true);
  size_t n = prefs.getBytes("cfg", &cfg, sizeof(cfg));
  prefs.end();
  if (n != sizeof(cfg) || cfg.magic != CFG_MAGIC) {
    defaultConfig();
    return false;
  }
  return true;
}

bool saveConfig() {
  prefs.begin("humanoide", false);
  size_t n = prefs.putBytes("cfg", &cfg, sizeof(cfg));
  prefs.end();
  cfgDirty = n != sizeof(cfg);
  return !cfgDirty;
}

int8_t findJoint(const String &name) {
  for (int8_t j = 0; j < NUM_JOINTS; j++) {
    if (name.equalsIgnoreCase(JOINT_NAMES[j])) return j;
  }
  return -1;
}

int8_t channelOfJoint(int8_t joint) {
  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    if (cfg.servo[i].joint == joint) return i;
  }
  return -1;
}

float clampToLimits(uint8_t ch, float angleDeg) {
  return constrain(angleDeg, cfg.servo[ch].minDeg, cfg.servo[ch].maxDeg);
}

float jointToServo(uint8_t ch, float jointDeg) {
  return cfg.servo[ch].zeroDeg + cfg.servo[ch].dir * jointDeg;
}

float servoToJoint(uint8_t ch, float servoDeg) {
  return (servoDeg - cfg.servo[ch].zeroDeg) * cfg.servo[ch].dir;
}

// ---------- Conversao angulo -> contagem 12 bits ----------
int angleToTicks(float angleDeg) {
  angleDeg = constrain(angleDeg, 0.0f, 180.0f);
  long scaled = (long)(angleDeg * 100.0f);          // preserva 2 casas decimais
  return (int)map(scaled, 0, 18000, SERVOMIN, SERVOMAX);
}

void applyAngle(uint8_t ch, float angleDeg) {
  channels[ch].currentAngle = angleDeg;
  if (cfg.servo[ch].enabled) pwm.setPWM(ch, 0, angleToTicks(angleDeg));
}

void disableOutput(uint8_t ch) {
  channels[ch].moving = false;
  channels[ch].wiggle = 0;
  pwm.setPWM(ch, 0, 4096);   // bit 12 = sempre desligado (servo solto)
}

// Interpolacao suave (ease-in/ease-out senoidal), cf. motion-kinematics.md
float easedAngle(float startDeg, float targetDeg, unsigned long elapsed, unsigned long total) {
  if (elapsed >= total) return targetDeg;
  float progress = (float)elapsed / (float)total;
  float factor = (1.0f - cos(PI * progress)) / 2.0f;
  return startDeg + (targetDeg - startDeg) * factor;
}

bool anyMoving() {
  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    if (channels[i].moving) return true;
  }
  return false;
}

// Unico caminho para mover um canal: aplica limites (fora do modo CFG) e velocidade maxima.
void moveChannel(uint8_t ch, float angleDeg, unsigned long durationMs) {
  if (!cfg.servo[ch].enabled) return;
  angleDeg = constrain(angleDeg, 0.0f, 180.0f);
  if (!cfgMode) angleDeg = clampToLimits(ch, angleDeg);
  float vel = cfgMode ? CFG_VEL_DPS : cfg.servo[ch].maxVel;
  if (cfgMode && cfg.servo[ch].maxVel > 0) vel = min(vel, cfg.servo[ch].maxVel);
  if (vel > 0) {
    unsigned long minMs = (unsigned long)(fabs(angleDeg - channels[ch].currentAngle) / vel * 1000.0f);
    durationMs = max(durationMs, minMs);
  } else if (durationMs <= STREAM_MS) {
    durationMs = 0;   // fluxo do controlador (JALL a 50 Hz): aplica direto, o servo suaviza
  }
  // Novo alvo com o canal ja em movimento (fluxo continuo) segue linear: o ease
  // reiniciaria com velocidade zero a cada quadro e o servo quase nao andaria.
  channels[ch].eased = !channels[ch].moving;
  channels[ch].startAngle = channels[ch].currentAngle;
  channels[ch].targetAngle = angleDeg;
  channels[ch].t0 = millis();
  channels[ch].duration = durationMs;
  if (durationMs == 0) {
    channels[ch].moving = false;
    applyAngle(ch, angleDeg);
  } else {
    channels[ch].moving = true;
  }
}

// Comando em graus de junta: satura nos limites e marca o canal como saturado.
void moveJoint(uint8_t ch, float jointDeg, unsigned long durationMs) {
  float servoDeg = jointToServo(ch, jointDeg);
  float lim = clampToLimits(ch, servoDeg);
  channels[ch].clamped = fabs(lim - servoDeg) > 0.01f;
  moveChannel(ch, lim, durationMs);
}

void updateServos(unsigned long now) {
  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    ChannelMove &c = channels[i];
    if (!c.moving) continue;
    unsigned long elapsed = now - c.t0;
    applyAngle(i, c.eased ? easedAngle(c.startAngle, c.targetAngle, elapsed, c.duration)
                          : c.startAngle + (c.targetAngle - c.startAngle) * min(1.0f, (float)elapsed / c.duration));
    if (elapsed >= c.duration) {
      c.moving = false;
      applyAngle(i, c.targetAngle);
      if (c.wiggle > 0) {   // proxima fase do WIGGLE: +, -, +, -, ..., centro
        c.wiggle--;
        float alvo = c.wiggle == 0 ? c.wiggleCenter
                   : c.wiggleCenter + ((c.wiggle % 2) ? -WIGGLE_DEG : WIGGLE_DEG);
        moveChannel(i, constrain(alvo, 0.0f, 180.0f), 250);
      }
    }
  }
}

// ---------- MPU-6050 ----------
uint8_t mpuAddr = 0;
bool imuOk = false;

struct ImuData {
  bool ok = false;
  float ax = 0.0f;     // g (±2g)
  float ay = 0.0f;
  float az = 0.0f;
  float gx = 0.0f;     // deg/s (±250 deg/s)
  float gy = 0.0f;
  float gz = 0.0f;
  float temp = 0.0f;   // °C
  float pitch = 0.0f;  // deg
  float roll = 0.0f;   // deg
};

void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(mpuAddr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool initMPU() {
  uint8_t candidates[2] = {MPU_ADDR_PRIMARY, MPU_ADDR_SECONDARY};
  for (uint8_t a : candidates) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      mpuAddr = a;
      // Acorda o MPU-6050 e seleciona PLL com giroscopio X como clock (0x01)
      Wire.beginTransmission(mpuAddr);
      Wire.write(MPU_REG_PWR_MGMT_1);
      Wire.write(0x01);
      if (Wire.endTransmission() != 0) continue;
      mpuWrite(MPU_REG_CONFIG, 0x03);     // DLPF ~44 Hz (vibracao dos servos)
      mpuWrite(MPU_REG_GYRO_CFG, 0x00);   // +/- 250 deg/s (131 LSB / deg/s)
      mpuWrite(MPU_REG_ACCEL_CFG, 0x00);  // +/- 2g (16384 LSB / g)
      imuOk = true;
      return true;
    }
  }
  imuOk = false;
  return false;
}

ImuData readMPU() {
  ImuData data;
  if (!imuOk) {
    if (!initMPU()) return data;
  }

  Wire.beginTransmission(mpuAddr);
  Wire.write(MPU_REG_ACCEL_XOUT);
  if (Wire.endTransmission(false) != 0) {
    imuOk = false;
    return data;
  }
  if (Wire.requestFrom((int)mpuAddr, 14) != 14) {
    imuOk = false;
    return data;
  }

  int16_t rawAx = (int16_t)((Wire.read() << 8) | Wire.read());
  int16_t rawAy = (int16_t)((Wire.read() << 8) | Wire.read());
  int16_t rawAz = (int16_t)((Wire.read() << 8) | Wire.read());
  int16_t rawTemp = (int16_t)((Wire.read() << 8) | Wire.read());
  int16_t rawGx = (int16_t)((Wire.read() << 8) | Wire.read());
  int16_t rawGy = (int16_t)((Wire.read() << 8) | Wire.read());
  int16_t rawGz = (int16_t)((Wire.read() << 8) | Wire.read());

  data.ok = true;
  data.ax = (float)rawAx / 16384.0f;
  data.ay = (float)rawAy / 16384.0f;
  data.az = (float)rawAz / 16384.0f;
  data.temp = ((float)rawTemp / 340.0f) + 36.53f;
  data.gx = (float)rawGx / 131.0f;
  data.gy = (float)rawGy / 131.0f;
  data.gz = (float)rawGz / 131.0f;
  data.pitch = atan2(-data.ax, sqrt(data.ay * data.ay + data.az * data.az)) * 180.0f / PI;
  data.roll = atan2(data.ay, data.az) * 180.0f / PI;
  return data;
}

void printImuJson(const ImuData &imu) {
  if (!imu.ok) {
    out->print("{\"ok\":false}");
    return;
  }
  out->print("{\"ok\":true,\"addr\":\"0x"); out->print(mpuAddr, HEX); out->print("\"");
  out->print(",\"ax\":"); out->print(imu.ax, 2);
  out->print(",\"ay\":"); out->print(imu.ay, 2);
  out->print(",\"az\":"); out->print(imu.az, 2);
  out->print(",\"gx\":"); out->print(imu.gx, 1);
  out->print(",\"gy\":"); out->print(imu.gy, 1);
  out->print(",\"gz\":"); out->print(imu.gz, 1);
  out->print(",\"temp\":"); out->print(imu.temp, 1);
  out->print(",\"pitch\":"); out->print(imu.pitch, 1);
  out->print(",\"roll\":"); out->print(imu.roll, 1);
  out->print("}");
}

// Amostra bruta para o equilibrio no host: I:<t_ms>:<ax>:<ay>:<az>:<gx>:<gy>:<gz>  (g e graus/s)
void sendImuSample() {
  ImuData imu = readMPU();
  if (!imu.ok) return;
  imuOut->print("I:"); imuOut->print(millis());
  imuOut->print(":"); imuOut->print(imu.ax, 3);
  imuOut->print(":"); imuOut->print(imu.ay, 3);
  imuOut->print(":"); imuOut->print(imu.az, 3);
  imuOut->print(":"); imuOut->print(imu.gx, 2);
  imuOut->print(":"); imuOut->print(imu.gy, 2);
  imuOut->print(":"); imuOut->println(imu.gz, 2);
}

// ---------- Respostas JSON ----------
void sendStatus() {
  out->print("{\"t\":"); out->print(millis());
  out->print(",\"mode\":\""); out->print(cfgMode ? "cfg" : "run");
  out->print("\",\"servos_busy\":"); out->print(anyMoving() ? "true" : "false");
  out->print(",\"channels\":[");
  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    if (i > 0) out->print(",");
    out->print((int)round(channels[i].currentAngle));
  }
  out->print("],\"joints\":{");
  bool first = true;
  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    if (cfg.servo[i].joint < 0) continue;
    if (!first) out->print(",");
    first = false;
    out->print("\""); out->print(JOINT_NAMES[cfg.servo[i].joint]); out->print("\":");
    out->print(servoToJoint(i, channels[i].currentAngle), 1);
  }
  out->print("},\"clamped\":[");
  first = true;
  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    if (!channels[i].clamped) continue;
    if (!first) out->print(",");
    first = false;
    out->print(i);
  }
  out->print("],\"imu\":");
  printImuJson(readMPU());
  out->println("}");
}

void sendConfig() {
  out->print("{\"mode\":\""); out->print(cfgMode ? "cfg" : "run");
  out->print("\",\"dirty\":"); out->print(cfgDirty ? "true" : "false");
  out->print(",\"servos\":[");
  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    const ServoCfg &s = cfg.servo[i];
    if (i > 0) out->print(",");
    out->print("{\"ch\":"); out->print(i);
    out->print(",\"joint\":");
    if (s.joint < 0) out->print("null");
    else { out->print("\""); out->print(JOINT_NAMES[s.joint]); out->print("\""); }
    out->print(",\"min\":"); out->print(s.minDeg, 1);
    out->print(",\"max\":"); out->print(s.maxDeg, 1);
    out->print(",\"zero\":"); out->print(s.zeroDeg, 1);
    out->print(",\"dir\":"); out->print(s.dir);
    out->print(",\"vel\":"); out->print(s.maxVel, 1);
    out->print(",\"en\":"); out->print(s.enabled ? 1 : 0);
    out->print(",\"pos\":"); out->print(channels[i].currentAngle, 1);
    out->print("}");
  }
  out->println("]}");
}

void printHelp() {
  out->println("CMDS: POS:<ch>:<ang>:<ms> J:<junta>:<graus>:<ms> JALL:<ms>:<g0>..<g11> CENTER[:<ch>] REST");
  out->println("      STATUS IMU IMUSTREAM:<ms> SCAN ON OFF PING INFO JOINTS CFG MODE:CFG|RUN HELP");
  out->println("CFG:  MAP:<ch>:<junta|-> LIM:<ch>:<min>:<max> SETMIN:<ch> SETMAX:<ch> SETZERO:<ch>");
  out->println("      ZERO:<ch>:<ang> DIR:<ch>:<1|-1> VEL:<ch>:<dps> EN:<ch>:<0|1> WIGGLE:<ch> SAVE LOAD DEFAULTS");
}

// ---------- Parser ----------
bool isNumber(const String &s) {
  if (s.length() == 0 || s.length() > 10) return false;
  bool dot = false, digit = false;
  for (uint8_t i = 0; i < s.length(); i++) {
    char c = s.charAt(i);
    if (c == '-' && i == 0) continue;
    if (c == '.' && !dot) { dot = true; continue; }
    if (!isDigit(c)) return false;
    digit = true;
  }
  return digit;
}

bool isInteger(const String &s) {
  return isNumber(s) && s.indexOf('.') < 0;
}

uint8_t split(const String &line, String tok[]) {
  uint8_t n = 0;
  int ini = 0;
  while (n < MAX_TOKENS) {
    int p = line.indexOf(':', ini);
    tok[n++] = p < 0 ? line.substring(ini) : line.substring(ini, p);
    if (p < 0) break;
    ini = p + 1;
  }
  return n;
}

// Le o canal do token; responde o erro e devolve -1 se invalido.
int parseChannel(const String &s) {
  if (!isInteger(s)) { out->println("ERR:INVALID_CMD"); return -1; }
  long ch = s.toInt();
  if (ch < 0 || ch >= NUM_CHANNELS) { out->println("ERR:OUT_OF_BOUNDS"); return -1; }
  return (int)ch;
}

bool requireCfg() {
  if (!cfgMode) out->println("ERR:NEED_CFG_MODE");
  return cfgMode;
}

void cmdPos(String tok[], uint8_t n) {
  if (n != 4 || !isNumber(tok[2]) || !isInteger(tok[3])) { out->println("ERR:INVALID_CMD"); return; }
  int ch = parseChannel(tok[1]);
  if (ch < 0) return;
  float ang = tok[2].toFloat();
  long dur = tok[3].toInt();
  if (ang < 0 || ang > 180 || dur < 0 || dur > 30000) { out->println("ERR:OUT_OF_BOUNDS"); return; }
  if (!cfg.servo[ch].enabled) { out->println("ERR:DISABLED"); return; }
  if (!cfgMode && (ang < cfg.servo[ch].minDeg || ang > cfg.servo[ch].maxDeg)) {
    out->print("ERR:LIMIT:"); out->print(cfg.servo[ch].minDeg, 1);
    out->print(":"); out->println(cfg.servo[ch].maxDeg, 1);
    return;
  }
  channels[ch].wiggle = 0;
  moveChannel(ch, ang, dur);
  out->println("ACK");
}

void cmdJoint(String tok[], uint8_t n) {
  if (cfgMode) { out->println("ERR:IN_CFG_MODE"); return; }
  if (n != 4 || !isNumber(tok[2]) || !isInteger(tok[3])) { out->println("ERR:INVALID_CMD"); return; }
  int8_t j = findJoint(tok[1]);
  if (j < 0) { out->println("ERR:UNKNOWN_JOINT"); return; }
  int8_t ch = channelOfJoint(j);
  if (ch < 0) { out->println("ERR:NOT_MAPPED"); return; }
  if (!cfg.servo[ch].enabled) { out->println("ERR:DISABLED"); return; }
  moveJoint(ch, tok[2].toFloat(), constrain(tok[3].toInt(), 0L, 30000L));
  out->println("ACK");
}

void cmdJointAll(String tok[], uint8_t n) {
  if (cfgMode) { out->println("ERR:IN_CFG_MODE"); return; }
  if (n != 2 + NUM_JOINTS || !isInteger(tok[1])) { out->println("ERR:INVALID_CMD"); return; }
  for (int8_t j = 0; j < NUM_JOINTS; j++) {
    const String &s = tok[2 + j];
    if (s != "X" && !isNumber(s)) { out->println("ERR:INVALID_CMD"); return; }
  }
  unsigned long dur = constrain(tok[1].toInt(), 0L, 30000L);
  for (int8_t j = 0; j < NUM_JOINTS; j++) {
    int8_t ch = channelOfJoint(j);
    if (ch < 0 || tok[2 + j] == "X") continue;   // juntas sem canal sao ignoradas
    moveJoint(ch, tok[2 + j].toFloat(), dur);
  }
  out->println("ACK");
}

void cmdConfig(const String &cmd, String tok[], uint8_t n) {
  if (!requireCfg()) return;
  if (cmd == "SAVE" || cmd == "LOAD" || cmd == "DEFAULTS") {
    if (cmd == "SAVE") {
      if (!saveConfig()) { out->println("ERR:NVS"); return; }
    } else if (cmd == "LOAD") {
      loadConfig();
      cfgDirty = false;
    } else {
      defaultConfig();
      cfgDirty = true;
    }
    out->println("ACK");
    return;
  }
  if (n < 2) { out->println("ERR:INVALID_CMD"); return; }
  int ch = parseChannel(tok[1]);
  if (ch < 0) return;
  ServoCfg &s = cfg.servo[ch];

  if (cmd == "MAP" && n == 3) {
    int8_t j = -1;
    if (tok[2] != "-") {
      j = findJoint(tok[2]);
      if (j < 0) { out->println("ERR:UNKNOWN_JOINT"); return; }
      int8_t antigo = channelOfJoint(j);
      if (antigo >= 0) cfg.servo[antigo].joint = -1;   // cada junta em um canal so
    }
    s.joint = j;
  } else if (cmd == "LIM" && n == 4 && isNumber(tok[2]) && isNumber(tok[3])) {
    float lo = tok[2].toFloat(), hi = tok[3].toFloat();
    if (lo < 0 || hi > 180 || lo >= hi) { out->println("ERR:OUT_OF_BOUNDS"); return; }
    s.minDeg = lo;
    s.maxDeg = hi;
  } else if ((cmd == "SETMIN" || cmd == "SETMAX" || cmd == "SETZERO") && n == 2) {
    float a = channels[ch].currentAngle;
    if (cmd == "SETMIN") {
      if (a >= s.maxDeg) { out->println("ERR:OUT_OF_BOUNDS"); return; }
      s.minDeg = a;
    } else if (cmd == "SETMAX") {
      if (a <= s.minDeg) { out->println("ERR:OUT_OF_BOUNDS"); return; }
      s.maxDeg = a;
    } else {
      s.zeroDeg = a;
    }
  } else if (cmd == "ZERO" && n == 3 && isNumber(tok[2])) {
    float a = tok[2].toFloat();
    if (a < 0 || a > 180) { out->println("ERR:OUT_OF_BOUNDS"); return; }
    s.zeroDeg = a;
  } else if (cmd == "DIR" && n == 3 && (tok[2] == "1" || tok[2] == "-1")) {
    s.dir = tok[2].toInt();
  } else if (cmd == "VEL" && n == 3 && isNumber(tok[2]) && tok[2].toFloat() >= 0) {
    s.maxVel = tok[2].toFloat();
  } else if (cmd == "EN" && n == 3 && (tok[2] == "0" || tok[2] == "1")) {
    s.enabled = tok[2] == "1";
    if (!s.enabled) disableOutput(ch);
    else applyAngle(ch, channels[ch].currentAngle);
  } else if (cmd == "WIGGLE" && n == 2) {
    if (!s.enabled) { out->println("ERR:DISABLED"); return; }
    channels[ch].wiggleCenter = channels[ch].currentAngle;
    channels[ch].wiggle = 6;
    moveChannel(ch, constrain(channels[ch].currentAngle + WIGGLE_DEG, 0.0f, 180.0f), 250);
    out->println("ACK");
    return;   // nao altera a configuracao
  } else {
    out->println("ERR:INVALID_CMD");
    return;
  }
  cfgDirty = true;
  out->println("ACK");
}

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;
  line.toUpperCase();   // nomes de junta sao comparados sem diferenciar maiusculas
  String tok[MAX_TOKENS];
  uint8_t n = split(line, tok);
  const String &cmd = tok[0];

  if (line == "PING") {
    out->println("PONG");
  } else if (line == "HELP") {
    printHelp();
    out->println("ACK");
  } else if (line == "STATUS") {
    sendStatus();
    out->println("ACK");
  } else if (line == "CFG") {
    sendConfig();
    out->println("ACK");
  } else if (line == "JOINTS") {
    for (int8_t j = 0; j < NUM_JOINTS; j++) {
      if (j > 0) out->print(",");
      out->print(JOINT_NAMES[j]);
    }
    out->println();
    out->println("ACK");
  } else if (line == "IMU") {
    printImuJson(readMPU());
    out->println();
    out->println("ACK");
  } else if (cmd == "IMUSTREAM" && n == 2 && isInteger(tok[1])) {
    long ms = tok[1].toInt();
    if (ms != 0 && (ms < 5 || ms > 1000)) { out->println("ERR:OUT_OF_BOUNDS"); return; }
    imuStreamMs = ms;
    imuOut = ms ? out : nullptr;   // as amostras vao pela porta que pediu
    out->println("ACK");
  } else if (line == "INFO") {
    out->print("INFO: fw=" FW_VERSION " mode=");
    out->print(cfgMode ? "cfg" : "run");
    out->print(" bt=");
#if USE_BT
    out->print("on name=" BT_NAME " mac=");
    out->print(SerialBT.getBtAddressString());
    out->print(" client=");
    out->println(SerialBT.hasClient() ? "yes" : "no");
#else
    out->println("off");
#endif
    out->println("ACK");
  } else if (cmd == "MODE" && n == 2 && (tok[1] == "CFG" || tok[1] == "RUN")) {
    cfgMode = tok[1] == "CFG";
    if (!cfgMode) {   // ao sair do CFG, traz quem ficou fora dos limites de volta
      for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
        channels[i].wiggle = 0;
        float a = clampToLimits(i, channels[i].currentAngle);
        if (a != channels[i].currentAngle) moveChannel(i, a, 0);
      }
    }
    out->println("ACK");
  } else if (line == "REST" || line == "GESTURE:REST") {
    for (uint8_t i = 0; i < NUM_CHANNELS; i++) moveChannel(i, cfg.servo[i].zeroDeg, 1500);
    out->println("ACK");
  } else if (line == "CENTER") {
    for (uint8_t i = 0; i < NUM_CHANNELS; i++) moveChannel(i, 90.0f, 1000);
    out->println("ACK");
  } else if (cmd == "CENTER" && n == 2) {
    int ch = parseChannel(tok[1]);
    if (ch < 0) return;
    moveChannel(ch, 90.0f, 1000);
    out->println("ACK");
  } else if (cmd == "POS") {
    cmdPos(tok, n);
  } else if (cmd == "J") {
    cmdJoint(tok, n);
  } else if (cmd == "JALL") {
    cmdJointAll(tok, n);
  } else if (cmd == "GESTURE") {
    out->println("ERR:NOT_MAPPED");   // gestos nomeados ficam no host (controle/controladores.py)
  } else if (cmd == "MAP" || cmd == "LIM" || cmd == "SETMIN" || cmd == "SETMAX" || cmd == "SETZERO" ||
             cmd == "ZERO" || cmd == "DIR" || cmd == "VEL" || cmd == "EN" || cmd == "WIGGLE" ||
             cmd == "SAVE" || cmd == "LOAD" || cmd == "DEFAULTS") {
    cmdConfig(cmd, tok, n);
  } else if (line == "SCAN") {
    for (uint8_t addr = 0x08; addr < 0x78; addr++) {
      Wire.beginTransmission(addr);
      if (Wire.endTransmission() == 0) {
        out->print("I2C: 0x");
        if (addr < 0x10) out->print("0");
        out->println(addr, HEX);
      }
    }
    out->println("ACK");
  } else if (line == "ON") {
#if USE_OE
    digitalWrite(OE_PIN, LOW);
    out->println("ACK");
#else
    out->println("ERR:INVALID_CMD");
#endif
  } else if (line == "OFF") {
#if USE_OE
    digitalWrite(OE_PIN, HIGH);
    out->println("ACK");
#else
    out->println("ERR:INVALID_CMD");
#endif
  } else {
    out->println("ERR:INVALID_CMD");
  }
}

void setup() {
  Serial.begin(BAUD_RATE);

#if USE_OE
  pinMode(OE_PIN, OUTPUT);
  digitalWrite(OE_PIN, HIGH);   // saidas desabilitadas durante o boot
#endif

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(400000);
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);

  initMPU();
  bool cfgOk = loadConfig();

  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    float a = clampToLimits(i, cfg.servo[i].zeroDeg);
    channels[i].currentAngle = channels[i].targetAngle = a;
    if (!cfg.servo[i].enabled) disableOutput(i);
    else if (CENTER_ON_BOOT) pwm.setPWM(i, 0, angleToTicks(a));
  }

#if USE_OE
  digitalWrite(OE_PIN, LOW);    // habilita as saidas apos init seguro
#endif

#if USE_BT
  SerialBT.setPin(BT_PIN, 4);
  if (SerialBT.begin(BT_NAME)) {
    Serial.print("BT: pronto, nome=" BT_NAME " mac=");
    Serial.println(SerialBT.getBtAddressString());
  } else {
    Serial.println("BT: falha ao iniciar (USB continua funcionando)");
  }
#endif

  Serial.println(cfgOk ? "CFG: carregada da flash" : "CFG: padrao (nada gravado ainda)");
  Serial.println("READY esp32-humanoide v" FW_VERSION);
  printHelp();
}

// Le linhas de uma porta e responde na mesma porta.
void pumpInput(Stream &port, String &buf) {
  while (port.available()) {
    char c = (char)port.read();
    if (c == '\n') {
      out = &port;
      handleLine(buf);
      out = &Serial;
      buf = "";
    } else if (c != '\r') {
      if (buf.length() < CMD_LINE_MAX) buf += c;
    }
  }
}

void loop() {
  pumpInput(Serial, rxLine);
#if USE_BT
  pumpInput(SerialBT, rxLineBT);
#endif

  unsigned long now = millis();
  if (now - lastServoUpdate >= SERVO_UPDATE_MS) {
    lastServoUpdate = now;
    updateServos(now);
  }
  if (now - lastTelemetry >= TELEMETRY_MS) {
    lastTelemetry = now;
    sendStatus();
  }
  if (imuStreamMs > 0 && imuOut && now - lastImuStream >= imuStreamMs) {
    lastImuStream = now;
    sendImuSample();
  }
}
