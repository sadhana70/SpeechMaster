
import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []

index = count()


def animate(i):
    data = pd.read_csv(r'C:\Users\Acer\codes\SpeechMaster\live_data.csv')
    x = data['x']
    y1 = data['pitch']
    y2 = data['loudness']

    plt.cla()

    plt.plot(x, y1, label='Pitch')
    plt.plot(x, y2, label='Loudness')

    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()