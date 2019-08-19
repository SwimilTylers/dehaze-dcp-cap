import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 255)
x = x.reshape((len(x), 1))

buf = np.arange(240, -1, -40).tolist()
yticks_range = [255]
for i in buf:
    yticks_range.append(i)

# min = max
t_d = 1.00855*np.exp(-0.121779-0.003764*x)
t_h = 1-0.003725*x

plt.figure()
plt.title("when max = min")
plt.plot(t_d, 'r')
plt.plot(t_h, 'k')
plt.legend(["cap", "dcp"])
plt.show()

sky = np.arange(255, 199, -1)
sky = sky.reshape((len(sky), 1))
r_sky = sky
atm = 255
for t in np.arange(0.1, 1.1, 0.01):
    buf = np.maximum(np.minimum((sky-atm)/t+atm, 255), 0)
    if r_sky is None:
        r_sky = buf
    else:
        r_sky = np.hstack((r_sky, buf))
plt.figure()
#plt.subplot(1, 2, 1)
#plt.imshow(sky/255, cmap="gray")
#plt.yticks([0, 20, 40, 55], [255, 235, 215, 200])
#plt.xticks([])
#plt.subplot(1, 2, 2)
plt.imshow(r_sky, vmin=0, cmap="gray")
plt.title("recovered intensity, A="+str(atm))
plt.ylabel("original intensity")
plt.yticks([0, 20, 40, 55], [255, 235, 215, 200])
plt.xlabel("transmission")
plt.xticks([1, 21, 41, 61, 81, 100], [0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.show()


# min = 0
t_d, t_h = None, None
for alpha in np.arange(0, 1.005, 0.005):
    buf = np.minimum(1.00855 * np.exp(0.658466 - 0.003764 * x - 0.780245 * alpha), 1)
    if t_d is None:
        t_d = buf
    else:
        t_d = np.hstack((t_d, buf))
    buf = 1-0.003725*alpha*x
    if t_h is None:
        t_h = buf
    else:
        t_h = np.hstack((t_h, buf))

plt.figure()
plt.title("cap transmission")
plt.imshow(t_d, cmap="gray")
plt.ylabel("max component")
plt.yticks(yticks_range)
plt.xlabel("min/max ratio")
plt.xticks([0, 40, 80, 120, 160, 200], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.show()

plt.figure()
plt.imshow(t_h, cmap="gray")
plt.title("dcp transmission")
plt.ylabel("max component")
plt.yticks(yticks_range)
plt.xlabel("min/max ratio")
plt.xticks([0, 40, 80, 120, 160, 200], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.show()
