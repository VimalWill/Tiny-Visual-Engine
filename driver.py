import RPi.GPIO as GPIO 

class MotorDriver:
    def __init__(self, en1, en2, in1, in2, in3, in4) -> None:

        #pin init for the motor and en 
        self.en1 = en1 
        self.en2 = en2 
        self.in1 = in1 
        self.in2 = in2 
        self.in3 = in3 
        self.in4 = in4 

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.en1, GPIO.OUT)
        GPIO.setup(self.en2, GPIO.OUT)
        GPIO.setup(self.in1, GPIO.OUT)
        GPIO.setup(self.in2, GPIO.OUT)
        GPIO.setup(self.in3, GPIO.OUT)
        GPIO.setup(self.in4, GPIO.OUT)

        self.mp_1 = GPIO.PWM(self.en1, 1000)
        self.mp_2 = GPIO.PWM(self.en2, 1000)

    def run(self, val):
        self.mp_1.start(25)
        self.mp_2.start(25)
        if val:
            GPIO.output(self.in1, GPIO.HIGH)
            GPIO.output(self.in2, GPIO.LOW)

            GPIO.output(self.in3, GPIO.HIGH)
            GPIO.output(self.in4, GPIO.LOW)
        
        else:
            GPIO.output(self.in1, GPIO.LOW)
            GPIO.output(self.in2, GPIO.LOW)

            GPIO.output(self.in3, GPIO.LOW)
            GPIO.output(self.in4, GPIO.LOW)
            

