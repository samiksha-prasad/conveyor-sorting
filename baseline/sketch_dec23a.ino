#include <Servo.h>

// Initialize three servos
Servo servoRed;
Servo servoGreen;
Servo servoBlue;

void setup() {
  Serial.begin(9600);

  // Attach servos to pins
  servoRed.attach(2);
  servoGreen.attach(5);
  servoBlue.attach(6);

  // Set initial positions
  servoRed.write(0);
  servoGreen.write(0);
  servoBlue.write(0);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    // Move the corresponding servo based on the command
    if (command == 'R') {
      servoRed.write(90);  // Adjust angle for Red
      delay(5000);
      servoRed.write(0);   // Reset position
    } else if (command == 'G') {
      servoGreen.write(90);  // Adjust angle for Green
      delay(5000);
      servoGreen.write(0);   // Reset position
    } else if (command == 'B') {
      servoBlue.write(90);  // Adjust angle for Blue
      delay(5000);
      servoBlue.write(0);   // Reset position
    }
  }
}