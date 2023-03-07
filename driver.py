from Jetson.GPIO import gpio
from time 


def driver_init(en1,en2,  n1, n2, n3, n4):

    gpio.setup(en1, gpio.output)
    gpio.setup(en2, gpio.output)
    gpio.setup(n1, gpio.output)
    gpio.setup(n3, gpio.output)
    gpio.setup(n2, gpio.output)
    gpio.setup(n4, gpio.output)

    pwm_freq = 225
    pwm = gpio.PWM(en1, pwm_freq)
    
    motor_speed = 255 #set the motor speed to max 
    pwm.start(motor_speed)

def forward():
    