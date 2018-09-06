from scipy.io import loadmat
import numpy as np
import math
import matplotlib.pyplot as plt
from params import SUBJECT_LIST, CHANNEL_NAMES, SAVE_PATH, STATE_LIST

SAVE_PATH = SAVE_PATH / "psd"
HR_labels = ("HR", "HR mean")
LR_labels = ("LR", "LR mean")

plt.figure()
ax = plt.axes()


def compute(val, k):
    return math.log(val / (k + 1))


all_subs = []
for sub in SUBJECT_LIST:
    all_elecs = []
    for elec in CHANNEL_NAMES:
        data = []
        for state in STATE_LIST:
            filename = SAVE_PATH / "PSDs_{}_s{}_{}_1000_0.00.mat".format(
                state, sub, elec
            )
            data.append(loadmat(filename)["data"].mean(axis=0))
        all_states = np.asarray(data).mean(axis=0)
        all_elecs.append(all_states)
    all_subs.append(np.asarray(all_elecs).mean(axis=0))

all_subs = np.asarray(all_subs)
hdr = all_subs[18:]
hdr = [[compute(a, i) for a in dat] for i, dat in enumerate(hdr)]
ldr = all_subs[:18]
ldr = [[compute(a, i) for a in dat] for i, dat in enumerate(ldr)]

for i in range(len(hdr)):
    plt.plot(
        range(1, 46), hdr[i], color="peachpuff", label="_nolegend_" if i > 0 else "HR"
    )
    plt.plot(
        range(1, 46), ldr[i], color="skyblue", label="_nolegend_" if i > 0 else "LR"
    )
#     plt.plot([math.log(i) for i in range(1, 46)], hdr[i], color='peachpuff')
#     plt.plot([math.log(i) for i in range(1, 46)], ldr[i], color='skyblue')
# plt.plot([math.log(i) for i in range(1, 46)], np.mean(hdr, axis=0), color='red')
# plt.plot([math.log(i) for i in range(1, 46)], np.mean(ldr, axis=0), color='blue')
plt.plot(range(1, 46), np.mean(hdr, axis=0), color="red", label="HR mean")
plt.plot(range(1, 46), np.mean(ldr, axis=0), color="blue", label="LR mean")
plt.legend()
plt.xlim(1, 45)
plt.ylabel("Power Spectral Density")
plt.xlabel("Frequency")
plt.show()
