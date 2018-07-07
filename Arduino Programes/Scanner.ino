#include<Stepper.h>
const int stepsPerRevolution = 3600;
/*接线说明
 * 注意：由于所选继电器为固态交流继电器，所以在继电器状态变为关闭后，
 * 必须将输入端电位拉低后才能关闭继电器，但如果采用直流继电器则无需这么麻烦
 * 激光触发 3
 * 激光使能 4
 * 电机脉冲 9
 * 电机方向 11
 * 电机使能 10 
 */
Stepper myStepper = Stepper(stepsPerRevolution, 8, 9);//引脚8，9为脉冲输出引脚
String data = ""; //串口接收的数据
//0，1引脚为串口引脚占用
int switchLight=3;//激光触发
int enableLight = 4; //激光控制引脚
int enableStepper = 10; //电机使能引脚
int directStepper = 11; //电机方向引脚
int stepSpeed=40;

void setup() {
  //引脚设置输出模式
  pinMode(switchLight, OUTPUT);
  pinMode(enableLight, OUTPUT);
  pinMode(enableStepper, OUTPUT);
  pinMode(directStepper, OUTPUT);
  //默认状态
  digitalWrite(switchLight, HIGH);
  digitalWrite(enableLight, LOW);
  digitalWrite(enableStepper, LOW);
  digitalWrite(directStepper, LOW);
  //设置步进电机步进距
  myStepper.setSpeed(stepSpeed);
  //初始化串口程序
  Serial.begin(9600);//设置串口波特率
  while (Serial.read() >= 0) {}; //清除串口缓存
}

void loop() {
  data = "";//接收数据置空
  if (Serial.available() > 0) {
    delay(5);
    //获取串口数据
    if (Serial.read() == '<') {
      data = Serial.readStringUntil('>');
    } else {
      Serial.println("Data ERROR!!!");
    }
    //Serial.println(data);
  }
  //打开激光
  if (!data.compareTo("lightON")) {
    Serial.println("1");
    digitalWrite(switchLight, HIGH);
    digitalWrite(enableLight, HIGH);
  }
  //关闭激光
  if (!data.compareTo("lightOFF")) {
    Serial.println("1");
    digitalWrite(switchLight, LOW);
    digitalWrite(enableLight, LOW);
  }
  //步进
  if (!data.compareTo("StepOne")) {
    Serial.println("1");
    myStepper.step(2);
  }
  //旋转一周
  if (!data.compareTo("StepRun")) {
    Serial.println("1");
    myStepper.setSpeed(40);
    for (int j = 0; j <= 6400; j++) {
      myStepper.step(1);
    }
    myStepper.setSpeed(stepSpeed);
  }
  //方向负
  if (!data.compareTo("directRight")) {
    Serial.println("1");
    digitalWrite(directStepper, HIGH);
  }
  //方向正
  if (!data.compareTo("directLeft")) {
    Serial.println("1");
    digitalWrite(directStepper, LOW);
  }
  //脱机
  if (!data.compareTo("StepOFF")) {
    Serial.println("1");
    digitalWrite(enableStepper, HIGH);
  }
  //运行
  if (!data.compareTo("StepON")) {
    Serial.println("1");
    digitalWrite(enableStepper, LOW);
  }
}
