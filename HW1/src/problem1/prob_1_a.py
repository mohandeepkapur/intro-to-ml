import numpy as np
import numpy.random as rnds
import matplotlib.pyplot as plt
import random as rnd

rnd.seed(3333323)
np.random.seed(44)

truths = []
samples = []
numSamples = 10000
# prior probs - how likely each thing is before any data is observed
pH0 = 0.7
pH1 = 0.3
# pH0|X updated belief of how likely H0 is
# pH1|X same

# X conditioned on Y = 1
mu0 = np.array([-1, 1, -1, 1])
sigma0 = np.array([
    [2, -0.5, 0.3, 0],
    [-0.5, 1, -0.5, 0],
    [0.3, -0.5, 1, 0],
    [0, 0, 0, 2]
])

# X conditioned on Y = 2
mu1 = np.array([1, 1, 1, 1])
sigma1 = np.array([
    [1, 0.3, -0.2, 0],
    [0.3, 2, 0.3, 0],
    [-0.2, 0.3, 1, 0],
    [0, 0, 0, 3]
])

for i in range(numSamples):
    # choose which likelihood distribution (reality) sample will be generated from
    if rnd.random() <= pH0:
        # sample from relevant x|y distribution (prob -> output)
        truths.append(0)
        samples.append(np.random.multivariate_normal(mu0, sigma0))
    else:
        truths.append(1)
        samples.append(np.random.multivariate_normal(mu1, sigma1))

# (output -> prob)
def fXgivenY(x, y):
    if (y == 0):
        mu = mu0
        sigma = sigma0
    else:
        mu = mu1
        sigma = sigma1
    return np.exp( (-0.5*(x - mu).T) @ np.linalg.inv(sigma) @ (x - mu) ) \
        / ( ((2*np.pi)**(len(x)/2)) * np.sqrt(np.linalg.det(sigma)))

def decisionThresholdPerformanceMetrics(LLRs, lnTh, truths01):
    # make inferences
    decisions = LLRs >= lnTh
    # calculate performance metrics
    num_h1s = np.sum(truths01)
    num_h0s = np.sum(~truths01)
    truePosRate = np.sum(decisions & truths01) / num_h1s # 1 & 1 = 1
    trueNegRate = np.sum(~decisions & ~truths01) / num_h0s # 0! & 0! = 1
    falsePosRate = np.sum(decisions & ~truths01) / num_h0s # 1 & 0! = 1
    falseNegRate = np.sum(~decisions & truths01) / num_h1s # 0! & 1 = 1
    return [[truePosRate, trueNegRate, falsePosRate, falseNegRate]]

LLRs = []
for sample in samples:
    pxg1 = fXgivenY(sample, 1)
    pxg0 = fXgivenY(sample, 0)
    LLRs.append( np.log(pxg1) - np.log(pxg0) )

thresholds = np.linspace(0.001, 10.00, 5000)
results = []
truths01 = np.array(truths).astype(bool)
for i in range(len(thresholds)):
    results.extend(decisionThresholdPerformanceMetrics(LLRs, np.log(thresholds[i]), truths01))

minIndex = 0
for i in range(len(results)):
    if ((results[minIndex][2] * pH0) + (results[minIndex][3] * pH1)
            >= (results[i][2] * pH0) + (results[i][3] * pH1)):
        minIndex = i

falsePosRate = [point[2] for point in results]
truePosRate = [point[0] for point in results]

print("optimal threshold  (p-error smallest): " + str(thresholds[minIndex]))
print(f"true pos rate: {truePosRate[minIndex]}")
print(f"false pos rate: {falsePosRate[minIndex]}")

plt.scatter(falsePosRate, truePosRate)
plt.plot(results[minIndex][2], results[minIndex][0], 'o', color='red')
plt.xlabel('P(E|Y=0)')
plt.ylabel('P(X=1|Y=1)')
plt.title('ROC Curve')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(falsePosRate, truePosRate, thresholds, c='r', marker='o')
ax.set_xlabel('False Positive Prob')
ax.set_ylabel('True Positive Prob')
ax.set_zlabel('Threshold')
ax.set_title('Thresholds + Threshold Performances')
plt.show()