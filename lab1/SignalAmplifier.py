import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Task - 3 Analog Signal Amplifier
    # my variant
    a = 31

    # range
    x = np.linspace(-10, 10, 100)

    # functions of a signal
    y1 = (a * 0.01) * np.sin(x)
    y2 = ((a + 3) * 0.01) * np.sin(x)
    y3 = (a * 0.01) * np.cos(x)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    ax1 = axes[0, 0]
    ax1.plot(x, y1, color="cyan", label="Закон І: y(x) = (a * 0.01) * sin(x)")
    ax1.axhline(0, color="grey")
    ax1.axvline(0, color="grey")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid()

    ax2 = axes[0, 1]
    ax2.plot(x, y2, color="magenta", label="Закон ІІ: y(x) = ((a + 3) * 0.01)) * sin(x)")
    ax2.axhline(0, color="grey")
    ax2.axvline(0, color="grey")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.grid()

    ax3 = axes[1, 0]
    ax3.plot(x, y3, color="orange", label="Закон ІІІ: y(x) = (a * 0.01) * cos(x)")
    ax3.axhline(0, color="grey")
    ax3.axvline(0, color="grey")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.legend()
    ax3.grid()

    # all signals plot
    ax = axes[1, 1]
    ax.plot(x, y1, color="cyan", label="Закон І")
    ax.plot(x, y2, color="magenta", label="Закон ІІ")
    ax.plot(x, y3, color="orange", label="Закон ІІІ")
    ax.axhline(0, color="grey")
    ax.axvline(0, color="grey")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid()

    plt.suptitle("Епюри тестових сигналів")
    plt.tight_layout()
    plt.show()
