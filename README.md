# Intro
LQR experimentation with several examples

   Error          control               True state

---------- LRQ ----------- Robot ------------
  |                                  |
  |                                  |
  |------------- Sensor -------------|

      Estimated state

We provide different examples:
- basic: simple linear problem
- affine: affine dynamics
- random: additive gaussian noise
- inverted-pendulum: non linear dynamics
- two wheel: non linear dynamics

Additionally, for non linear dynamics, we provide an iterative LQR solver called iLQR
## Differential Drive Robot

Constants:
R_l : left wheel radius
R_r : right wheel radius
L   : wheel distance

State:
x_t :       x-coordinate of robot
y_t :       y-coordinate of robot
theta_t :   heading

Actions:
w_l :   left motor RPM
w_r :   right motor RPM


Dynamics:
x_t+1      = x_t     + PI * (w_r * R_r + w_l * R_l) * cos(theta_t) * dt
y_t+1      = y_t     + PI * (w_r * R_r + w_l * R_l) * sin(theta_t) * dt
theta_t+1  = theta_t + 2 * PI * (w_r * R_r - w_l * R_l)/L * dt

Linearization:
State_t+1 = A * state_t + B * actions_t

A = [
      1 0  0
      0 1  0
      0 0  1
    ]

B = [
      PI * R_r * cos(theta_t) * dt    PI * R_l * cos(theta_t) * dt
      PI * R_r * sin(theta_t) * dt    PI * R_l * sin(theta_t) * dt
      2* PI * R_r * dt                -2 * PI * R_l * dt
]


