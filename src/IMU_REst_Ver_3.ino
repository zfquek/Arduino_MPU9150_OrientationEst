#include <Wire.h>
#include <stlport.h>
#include <Eigen30.h>

using namespace Eigen;

// ================== Register values for the MPU9150 IMU ============================
#define ACC_DEVICE (0x53)            // Device address for the accelerometer
#define ACC_POWER_CTRL (0x2D)        // Accelerometer power control register address
#define ACC_DATA_FORMAT (0x31)       // Accelerometer data format register address
#define ACC_DATA_X0 (0x32)           // Accelerometer X axis Data Low Byte address
#define ACC_DATA_X1 (0x33)           // Accelerometer X axis Data High Byte address
#define ACC_DATA_Y0 (0x34)           // Accelerometer Y axis Data Low Byte address
#define ACC_DATA_Y1 (0x35)           // Accelerometer Y axis Data High Byte address
#define ACC_DATA_Z0 (0x36)           // Accelerometer Z axis Data Low Byte address
#define ACC_DATA_Z1 (0x37)           // Accelerometer Z axis Data High Byte address

#define GYRO_DEVICE (0x69)           // Device address for L3G4200D Gyroscope
#define GYRO_CTRL_REG1 (0x20)        // Gyroscope control register 1
#define GYRO_CTRL_REG2 (0x21)        // Gyroscope control register 2
#define GYRO_CTRL_REG3 (0x22)        // Gyroscope control register 3
#define GYRO_CTRL_REG4 (0x23)        // Gyroscope control register 4
#define GYRO_CTRL_REG5 (0x24)        // Gyroscope control register 5
#define GYRO_DATAXL (0x28)           // Gyroscope X axis Data Low Byte address
#define GYRO_DATAXH (0x29)           // Gyroscope X axis Data High Byte address
#define GYRO_DATAYL (0x2A)           // Gyroscope Y axis Data Low Byte address
#define GYRO_DATAYH (0x2B)           // Gyroscope Y axis Data High Byte address
#define GYRO_DATAZL (0x2C)           // Gyroscope Z axis Data Low Byte address
#define GYRO_DATAZH (0x2D)           // Gyroscope Z axis Data High Byte address

#define MAG_DEVICE (0x1E)            // Device address for HMC5883L Magnetometer
#define MAG_CONFIG_REG_A (0x00)      // Magnetometer configuration register A
#define MAG_CONFIG_REG_B (0x01)      // Magnetometer configuration register B
#define MAG_MODE_REG (0x02)          // Magnetometer mode register
#define MAG_DATA_XH (0x03)           // Magnetometer X Axis Data High Byte address
#define MAG_DATA_XL (0x04)           // Magnetometer X Axis Data Low Byte address
#define MAG_DATA_YH (0x05)           // Magnetometer Y Axis Data High Byte address
#define MAG_DATA_YL (0x06)           // Magnetometer Y Axis Data Low Byte address
#define MAG_DATA_ZH (0x07)           // Magnetometer Z Axis Data High Byte address
#define MAG_DATA_ZL (0x08)           // Magnetometer Z Axis Data Low Byte address

// ===================================================================================
#define SRSHORT_ACCDATA  0xFFFF
#define SRSHORT_GYRODATA 0xFFFE
#define SRSHORT_MAGDATA  0xFFFD
#define SRSHORT_ALLDATA  0xFFFC

#define ACC_BIT_RESOLUTION 4*9.81/1024
#define MAG_BIT_RESOLUTION 1
#define GYRO_BIT_RESOLUTION 2*250/65536*3.1421/180

#define ACC_X_CALIBRATION 9.81/10.0
#define ACC_Y_CALIBRATION 9.81/10.0
#define ACC_Z_CALIBRATION 9.81/9.3

#define MAG_X_CALIBRATION_SCALE
#define MAG_Y_CALIBRATION_SCALE
#define MAG_Z_CALIBRATION_SCALE

#define MAG_X_CALIBRATION_OFFSET  0
#define MAG_Y_CALIBRATION_OFFSET  255
#define MAG_Z_CALIBRATION_OFFSET  0


// ===================================================================================

int SensorType = 0;
int ConfigData = 0;
int State = 0;
char ConfigAddressByte = 0;
char ConfigDataByte = 0;
byte _buff[6];

// Accelerometer, magnetometer, and gyroscope vector values
Vector3i DataAccVec;
Vector3i DataMagVec;
Vector3i DataGyroVec;

// Accelerometer, magnetometer, and gyroscope vector values in double
Vector3d DataAccVecd;
Vector3d DataMagVecd;
Vector3d DataGyroVecd;

// Filtered accelerometer vector values
Vector3d DataAccVecd_Preflitered;
Vector3d DataAccVecd_Flitered;

// Components of the rotation matrix (x, y, and z direction of body frame relative to world frame)
Vector3d IVec_Current;
Vector3d JVec_Current;
Vector3d KVec_Current;
Matrix3d R_est;
Matrix3d R_y;

Quaterniond q_est;
Quaterniond dq_est;
Vector3d bhat;
Vector3d dbhat;
double kp = 10;
double ki = 1;

double Lambda = 0.03;
double Sigma = 0.94;
double accFilterAlpha = 1.00;

double t_current = 0;
double t_new = 0;

double dt;
int Debug_PrintR = 1;



void setup()
{
  Wire.begin();          // join i2c bus (address optional for master)

  // initialize serial:
  Serial.begin(115200);

  // Initialize sensor
  initAllSensors();

  // Perform an initial read
  readAllSensors();

  // Record the current time
  t_current = (double)millis();

  CalcRotation();
  R_est = R_y;

  // Calculate the quaternion from the estimated rotation matrix
  RotToQuaternion(q_est, R_est);
}

void loop()
{
  // Read the accelerometer, magnetometer, and the gyroscope signal
  readAllSensors();

  // Record the time
  t_new = (double)millis();

  // Calculate the change in time since last sensor read
  dt = (t_new - t_current) * 0.001;

  // Update the current time
  t_current = t_new;

  // Calculate the estimated rotation matrix based on the nonlinear complementary filter
  CalcRotation();

  // ---------------------------------------------------------------------------------------
  Matrix3d R_err;
  R_err = R_est.transpose() * R_y;

  Matrix3d R_antisymmetric = antiSymmetric(R_err);

  Vector3d omega = extractSkewSymmetric(antiSymmetric(R_err));

  dq_est = (q_est * omegaToQuaternion(- bhat + kp * omega));
  dq_est.w() = dq_est.w() * 0.5;
  dq_est.x() = dq_est.x() * 0.5;
  dq_est.y() = dq_est.y() * 0.5;
  dq_est.z() = dq_est.z() * 0.5;

  dbhat = -ki * omega;

  // ---------------------------------------------------------------------------------------
  // Perform the update
  q_est.w() = q_est.w() + dq_est.w() * dt;
  q_est.x() = q_est.x() + dq_est.x() * dt;
  q_est.y() = q_est.y() + dq_est.y() * dt;
  q_est.z() = q_est.z() + dq_est.z() * dt;
  q_est.normalize();

  bhat = bhat + dbhat * dt;

  R_est = q_est.toRotationMatrix();

  // Print the rotation matrix to the serial port
  if (Serial.available() > 0)
  {
    int val = Serial.read();
    if (val == 'R')
    {
      printRotationToMatlab(R_est);
    }
  }
  if (Debug_PrintR == 1)
  {
    //printQuaternion(q_est);
  }
}

void initAllSensors()
{
  // Initialization for accelerometer
  // Put the ADXL345 into +/- 2G range by writing the value 0x00 to the DATA_FORMAT register.
  writeTo(ACC_DEVICE, ACC_DATA_FORMAT, 0x00);
  // Put the ADXL345 into Measurement Mode by writing 0x08 to the POWER_CTL register.
  writeTo(ACC_DEVICE, ACC_POWER_CTRL, 0x08);

  // Initialization for gyroscope
  writeTo(GYRO_DEVICE, GYRO_CTRL_REG1, 0b00001111);
  writeTo(GYRO_DEVICE, GYRO_CTRL_REG2, 0b00000000);
  writeTo(GYRO_DEVICE, GYRO_CTRL_REG3, 0b00001000);
  writeTo(GYRO_DEVICE, GYRO_CTRL_REG4, 0b00000000);
  writeTo(GYRO_DEVICE, GYRO_CTRL_REG5, 0b00000000);

  // Initialization for magnetometer
  writeTo(MAG_DEVICE, MAG_CONFIG_REG_A, 0x74);
  writeTo(MAG_DEVICE, MAG_CONFIG_REG_B, 0x20);
  writeTo(MAG_DEVICE, MAG_MODE_REG, 0x00);
}

void readAllSensors()
{
  // Read all sensors
  readAccel();
  readGyro();
  readMag();
}

void readAccel()
{
  uint8_t howManyBytesToRead = 6;
  readFrom(ACC_DEVICE, ACC_DATA_X0, howManyBytesToRead, _buff); //read the acceleration data from the ADXL345

  // each axis reading comes in 10 bit resolution, ie 2 bytes with LSB first
  // thus we are converting both bytes in to one int
  DataAccVec[0] = (((int)_buff[1]) << 8) | _buff[0];
  DataAccVec[1] = (((int)_buff[3]) << 8) | _buff[2];
  DataAccVec[2] = (((int)_buff[5]) << 8) | _buff[4];

  // Calculate the physical acceleration vector in the body coordinate
  DataAccVecd_Preflitered[0] = -((double)DataAccVec[0] * (double)ACC_BIT_RESOLUTION * (double)ACC_X_CALIBRATION);
  DataAccVecd_Preflitered[1] = -((double)DataAccVec[1] * (double)ACC_BIT_RESOLUTION * (double)ACC_Y_CALIBRATION);
  DataAccVecd_Preflitered[2] = -((double)DataAccVec[2] * (double)ACC_BIT_RESOLUTION * (double)ACC_Z_CALIBRATION);

  // Perform low pass filtering on the accelerometer signal
  DataAccVecd_Flitered[0] = accFilterAlpha * DataAccVecd_Preflitered[0] + (1 - accFilterAlpha) * DataAccVecd_Flitered[0];
  DataAccVecd_Flitered[1] = accFilterAlpha * DataAccVecd_Preflitered[1] + (1 - accFilterAlpha) * DataAccVecd_Flitered[1];
  DataAccVecd_Flitered[2] = accFilterAlpha * DataAccVecd_Preflitered[2] + (1 - accFilterAlpha) * DataAccVecd_Flitered[2];

  // Normalize the accelerometer signal
  double Norm = sqrt(pow(DataAccVecd_Flitered[0], 2) + pow(DataAccVecd_Flitered[1], 2) + pow(DataAccVecd_Flitered[2], 2));
  DataAccVecd[0] = DataAccVecd_Flitered[0] / Norm;
  DataAccVecd[1] = DataAccVecd_Flitered[1] / Norm;
  DataAccVecd[2] = DataAccVecd_Flitered[2] / Norm;
}

void readGyro()
{
  readFrom(GYRO_DEVICE, (GYRO_DATAXL | 0x80), 6, _buff); // read the gyroscope data from L3G4200D gyroscope

  // Data is in 2's complement
  DataGyroVec[0] = ((_buff[1] & 0xFF) << 8) | (_buff[0] & 0xFF);
  DataGyroVec[1] = ((_buff[3] & 0xFF) << 8) | (_buff[2] & 0xFF);
  DataGyroVec[2] = ((_buff[5] & 0xFF) << 8) | (_buff[4] & 0xFF);

  // Calculate the physical angular velocity vector in the body coordinate
  DataGyroVecd[0] = (double)DataGyroVec[0] * (double)GYRO_BIT_RESOLUTION;
  DataGyroVecd[1] = (double)DataGyroVec[1] * (double)GYRO_BIT_RESOLUTION;
  DataGyroVecd[2] = (double)DataGyroVec[2] * (double)GYRO_BIT_RESOLUTION;
}

void readMag()
{
  uint8_t howManyBytesToRead = 6;
  readFrom(MAG_DEVICE, MAG_DATA_XH, howManyBytesToRead, _buff); //read the magnetometer data from the ADXL345

  DataMagVec[0] = ((_buff[0] & 0xFF) << 8) | (_buff[1] & 0xFF);
  DataMagVec[2] = ((_buff[2] & 0xFF) << 8) | (_buff[3] & 0xFF);
  DataMagVec[1] = ((_buff[4] & 0xFF) << 8) | (_buff[5] & 0xFF);

  // Calculate the physical magnetic field vector in the body coordinate
  DataMagVecd[0] = (double)DataMagVec[0] * (double)MAG_BIT_RESOLUTION + (double)MAG_X_CALIBRATION_OFFSET;
  DataMagVecd[1] = (double)DataMagVec[1] * (double)MAG_BIT_RESOLUTION + (double)MAG_Y_CALIBRATION_OFFSET;
  DataMagVecd[2] = (double)DataMagVec[2] * (double)MAG_BIT_RESOLUTION + (double)MAG_Z_CALIBRATION_OFFSET;

  // Obtain the component of the magnetic field vector that is parallel to the gravity vector
  double MagDotAcc = DataAccVecd[0] * DataMagVecd[0] + DataAccVecd[1] * DataMagVecd[1] + DataAccVecd[2] * DataMagVecd[2];

  // Remove the component parallel to the gravity vector from the magnetic field component
  DataMagVecd[0] = DataMagVecd[0] - MagDotAcc * DataAccVecd[0];
  DataMagVecd[1] = DataMagVecd[1] - MagDotAcc * DataAccVecd[1];
  DataMagVecd[2] = DataMagVecd[2] - MagDotAcc * DataAccVecd[2];

  // Normalize the magnetometer signal
  double Norm = sqrt(pow(DataMagVecd[0], 2) + pow(DataMagVecd[1], 2) + pow(DataMagVecd[2], 2));
  DataMagVecd[0] = DataMagVecd[0] / Norm;
  DataMagVecd[1] = DataMagVecd[1] / Norm;
  DataMagVecd[2] = DataMagVecd[2] / Norm;
}

void CalcRotation()
{
  // Calculate the initial rotation matrix of the body frame relative to the inertia frame
  for (int i = 0; i < 3; i++)
  {
    // Let the accelerometer reading be the z-axis
    KVec_Current[i] = DataAccVecd[i];

    // Let the magnetometer reading be the x-axis
    IVec_Current[i] = DataMagVecd[i];
  }

  // Use cross product to determine the y-axis
  JVec_Current[0] = KVec_Current[1] * IVec_Current[2] - KVec_Current[2] * IVec_Current[1];
  JVec_Current[1] = KVec_Current[2] * IVec_Current[0] - KVec_Current[0] * IVec_Current[2];
  JVec_Current[2] = KVec_Current[0] * IVec_Current[1] - KVec_Current[1] * IVec_Current[0];

  // Set the current rotation matrix estimatec
  R_y(0, 0) = IVec_Current[0]; R_y(0, 1) = JVec_Current[0]; R_y(0, 2) = KVec_Current[0];
  R_y(1, 0) = IVec_Current[1]; R_y(1, 1) = JVec_Current[1]; R_y(1, 2) = KVec_Current[1];
  R_y(2, 0) = IVec_Current[2]; R_y(2, 1) = JVec_Current[2]; R_y(2, 2) = KVec_Current[2];
}


void RotToQuaternion( Quaterniond& q, Matrix3d a)
{
  float trace = a(0, 0) + a(1, 1) + a(2, 2);

  if ( trace > 0 ) { // I changed M_EPSILON to 0
    double s = 0.5f / sqrtf(trace + 1.0f);
    q.w() = 0.25f / s;
    q.x() = ( a(2, 1) - a(1, 2) ) * s;
    q.y() = ( a(0, 2) - a(2, 0) ) * s;
    q.z() = ( a(1, 0) - a(0, 1) ) * s;
  }
  else
  {
    if ( a(0, 0) > a(1, 1) && a(0, 0) > a(2, 2) )
    {
      double s = 2.0f * sqrtf( 1.0f + a(0, 0) - a(1, 1) - a(2, 2));
      q.w() = (a(2, 1) - a(1, 2) ) / s;
      q.x() = 0.25f * s;
      q.y() = (a(0, 1) + a(1, 0) ) / s;
      q.z() = (a(0, 2) + a(2, 0) ) / s;
    }
    else if (a(1, 1) > a(2, 2))
    {
      double s = 2.0f * sqrtf( 1.0f + a(1, 1) - a(0, 0) - a(2, 2));
      q.w() = (a(0, 2) - a(2, 0) ) / s;
      q.x() = (a(0, 1) + a(1, 0) ) / s;
      q.y() = 0.25f * s;
      q.z() = (a(1, 2) + a(2, 1) ) / s;
    }
    else
    {
      double s = 2.0f * sqrtf( 1.0f + a(2, 2) - a(0, 0) - a(1, 1) );
      q.w() = (a(1, 0) - a(0, 1) ) / s;
      q.x() = (a(0, 2) + a(2, 0) ) / s;
      q.y() = (a(1, 2) + a(2, 1) ) / s;
      q.z() = 0.25f * s;
    }
  }
}

void printQuaternion(Quaterniond& q)
{
  Serial.print("q.w: ");
  Serial.print(q.w());
  Serial.print(" q.x: ");
  Serial.print(q.x());
  Serial.print(" q.y: ");
  Serial.print(q.y());
  Serial.print(" q.z: ");
  Serial.print(q.z());
  Serial.print(" q.mag: ");
  Serial.println(q.norm());
}

void printRotation(Matrix3d& R)
{
  Serial.print(" r00: ");
  Serial.print(R(0, 0));
  Serial.print(" r01: ");
  Serial.print(R(0, 1));
  Serial.print(" r02: ");
  Serial.print(R(0, 2));
  Serial.print(" r10: ");
  Serial.print(R(1, 0));
  Serial.print(" r11: ");
  Serial.print(R(1, 1));
  Serial.print(" r12: ");
  Serial.print(R(1, 2));
  Serial.print(" r20: ");
  Serial.print(R(2, 0));
  Serial.print(" r21: ");
  Serial.print(R(2, 1));
  Serial.print(" r22: ");
  Serial.println(R(2, 2));
}

void printRotationToMatlab(Matrix3d& R)
{
  double R_Values[9];
  R_Values[0] = R(0, 0);
  R_Values[1] = R(0, 1);
  R_Values[2] = R(0, 2);
  R_Values[3] = R(1, 0);
  R_Values[4] = R(1, 1);
  R_Values[5] = R(1, 2);
  R_Values[6] = R(2, 0);
  R_Values[7] = R(2, 1);
  R_Values[8] = R(2, 2);

  Serial.write((unsigned char*)R_Values, 36);
}

void printQuaternionToMatlab(Quaterniond q)
{
  float q_Values[4];
  q_Values[0] = q.w();
  q_Values[1] = q.x();
  q_Values[2] = q.y();
  q_Values[3] = q.z();

  Serial.write((char*)q_Values, 16);
}

void printVector(Vector3d& V)
{
  Serial.print("V[0]: ");
  Serial.print(V[0]);
  Serial.print(" V[1]: ");
  Serial.print(V[1]);
  Serial.print(" V[2]: ");
  Serial.println(V[2]);
}

Matrix3d antiSymmetric(Matrix3d M)
{
  return 0.5 * (M - M.transpose());
}

Vector3d extractSkewSymmetric(Matrix3d M)
{
  Vector3d t_returnVect;

  t_returnVect[0] = M(2, 1);
  t_returnVect[1] = M(0, 2);
  t_returnVect[2] = M(1, 0);

  return t_returnVect;
}

Quaterniond omegaToQuaternion(Vector3d w)
{
  Quaterniond q;
  q.w() = 0;
  q.x() = w[0];
  q.y() = w[1];
  q.z() = w[2];

  return q;
}


// Write to the I2C device
void writeTo(int device, byte address, byte val)
{
  Wire.beginTransmission(device);   // start transmission to device
  Wire.write(address);              // send register address
  Wire.write(val);                  // send value to write
  Wire.endTransmission();           // end transmission
}

// Reads num bytes from the I2C device starting from address register on device in to _buff array
void readFrom(int device, byte address, int num, byte _buff[])
{
  Wire.beginTransmission(device);   // start transmission to device
  Wire.write(address);              // sends address to read from
  Wire.endTransmission();           // end transmission

  Wire.beginTransmission(device);   // start transmission to device
  Wire.requestFrom(device, num);    // request 6 bytes from device

  int i = 0;
  while (Wire.available())          // device may send less than requested (abnormal)
  {
    _buff[i] = Wire.read();         // receive a byte
    i++;
  }
  Wire.endTransmission();           // end transmission
}

