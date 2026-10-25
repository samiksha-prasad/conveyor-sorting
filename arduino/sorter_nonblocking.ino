// Non-blocking variant of the sorter firmware (not tested on hardware).
//
// Same serial protocol as baseline/sketch_dec23a.ino ('R', 'G', 'B' at 9600 baud).
// Two changes from the baseline firmware:
//   1. No delay(): each diverter keeps its own timers, so the board keeps
//      reading commands while a diverter is open, and the three diverters
//      work independently.
//   2. Timed opening: a command arrives when the camera sees the object, but
//      the object reaches its diverter later. Each command is queued and the
//      diverter opens TRAVEL_MS later, for HOLD_MS, i.e. only while that
//      object is passing. A short window means a neighbouring object is
//      much less likely to be pushed into the wrong bin.
//
// Timing is counted from when the command arrives, which is when the
// classifier finishes, so classification must take well under HOLD_MS / 2.
//
// TRAVEL_MS depends on belt speed and the camera-to-diverter distance and
// must be measured on the real belt. Values here match the simulation in
// sim/conveyor.py.

#include <Servo.h>

const unsigned long HOLD_MS = 600;
const int QUEUE = 8;

struct Diverter {
  Servo servo;
  char command;
  int pin;
  unsigned long travelMs;
  unsigned long pending[QUEUE];  // scheduled opening times
  int count;
  bool open;
  unsigned long closeAt;
};

Diverter diverters[] = {
  {Servo(), 'R', 2, 2000, {0}, 0, false, 0},
  {Servo(), 'G', 5, 2500, {0}, 0, false, 0},
  {Servo(), 'B', 6, 3000, {0}, 0, false, 0},
};
const int N = sizeof(diverters) / sizeof(diverters[0]);

void setup() {
  for (int i = 0; i < N; i++) {
    diverters[i].servo.attach(diverters[i].pin);
    diverters[i].servo.write(0);
  }
  Serial.begin(9600);
}

void loop() {
  unsigned long now = millis();

  while (Serial.available() > 0) {
    char command = Serial.read();
    for (int i = 0; i < N; i++) {
      Diverter &d = diverters[i];
      if (d.command == command && d.count < QUEUE) {
        // centre the open window on the object's expected arrival
        d.pending[d.count++] = now + d.travelMs - HOLD_MS / 2;
      }
    }
  }

  for (int i = 0; i < N; i++) {
    Diverter &d = diverters[i];
    // open when the earliest queued object is due
    if (d.count > 0 && (long)(now - d.pending[0]) >= 0) {
      d.servo.write(90);
      d.open = true;
      d.closeAt = d.pending[0] + HOLD_MS;
      for (int j = 1; j < d.count; j++) d.pending[j - 1] = d.pending[j];
      d.count--;
    }
    if (d.open && (long)(now - d.closeAt) >= 0) {
      d.servo.write(0);
      d.open = false;
    }
  }
}
