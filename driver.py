import Jetson.GPIO as GPIO

class MotorDriver:
    def __init__(self, motor_pins, enable_pins):
        self.motor_pins = motor_pins
        self.enable_pins = enable_pins
        self.GPIO = GPIO

        # Set up GPIO pins
        self.GPIO.setmode(GPIO.BOARD)
        for pin in self.motor_pins + self.enable_pins:
            self.GPIO.setup(pin, self.GPIO.OUT)

        # Initialize PWM objects
        self.motors = []
        for pin in self.enable_pins:
            self.motors.append(self.GPIO.PWM(pin, 100))
            self.motors[-1].start(0)

    def set_direction(self, in1, in2, direction):
        if direction == "forward":
            self.GPIO.output(in1, self.GPIO.HIGH)
            self.GPIO.output(in2, self.GPIO.LOW)
        elif direction == "reverse":
            self.GPIO.output(in1, self.GPIO.LOW)
            self.GPIO.output(in2, self.GPIO.HIGH)
        else:
            raise ValueError("Invalid direction: {}".format(direction))

    def set_speed(self, speed):
        for motor in self.motors:
            motor.ChangeDutyCycle(speed)

    def start_stop(self, start):
        if start:
            for motor in self.motors:
                motor.start(50)
        else:
            for motor in self.motors:
                motor.stop()

    def cleanup(self):
        self.GPIO.cleanup()
