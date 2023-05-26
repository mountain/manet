import numpy as np
import matplotlib.pyplot as plt

h = 2.0
l = 10.0


def pfun(r, ts):
    omega = np.sqrt(9.81 / r)
    xs = r * (omega * ts - np.sin(omega * ts))
    ys = h - r * (1 - np.cos(omega * ts))
    return xs, ys


def seek_r(l):
    for r in np.linspace(200, 0.1, 5001):
        for t in np.linspace(0, 5, 1001):
            x, y = pfun(r, t)
            if np.abs(y) < 0.05 and np.abs(x - l) < 0.05:
                print(r, t)
                yield r, t


def plot(l):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for r, tmx in seek_r(l):
        ts = np.linspace(0, tmx, 1001)
        xs, ys = pfun(r, ts)
        ax.plot(xs, ys)

    plt.show()


# plot(l)

last_min = 1e-5
for r in np.linspace(1.789, 1.790, 10001):
    ts = np.linspace(0, 1.963, 100001)
    xs, ys = pfun(r, ts)
    loss = ys * ys + (xs - 10) * (xs - 10)
    idx = np.argmin(loss)
    if loss[idx] < last_min:
        print(r, ts[idx], loss[idx])
        last_min = loss[idx]

