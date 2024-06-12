import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
L1, L2 = 1.0, 1.0
m1, m2 = 1.0, 1.0
g = 9.8


def double_pendulum(t, y, L1, L2, m1, m2, g=9.81):
    theta1, omega1, theta2, omega2 = y
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    domega1 = (m2*g*np.sin(theta2)*c - m2*s*(L1*omega1**2*c + L2*omega2**2) -
               (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)

    domega2 = ((m1+m2)*(L1*omega1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c)
               + m2*L2*omega2**2*s*c) / L2 / (m1 + m2*s**2)

    dtheta1 = omega1
    dtheta2 = omega2
    return dtheta1, domega1, dtheta2, domega2


y0 = [np.pi / 2, 0, np.pi / 2, 0]
t_span = (0, 15)  # time span
t_eval = np.linspace(*t_span, 500)

sol = solve_ivp(double_pendulum, t_span, y0, args=(L1, L2, m1, m2, g),
                t_eval=t_eval, method='RK45')

# Calculate the positions of the pendulums
x1 = L2 * np.sin(sol.y[0])
y1 = -L1 * np.cos(sol.y[0])
x2 = x1 + L2 * np.sin(sol.y[2])
y2 = y1 - L2 * np.cos(sol.y[2])

fig, ax = plt.subplots()
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
line, = ax.plot([], [], 'o-', lw=2)


def animate(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    return line,


ani = FuncAnimation(fig, animate, frames=len(t_eval), interval=25, blit=True)
plt.show()
