import numpy as np
import random as rnd
import matplotlib.pyplot as plt

num_class = 3
np.random.seed(20020)

# PART B
# not assuming 0-1 loss, so decision rule will change a bit from MAP

loss_matrix_10 = np.array([[0, 1, 10],
                           [1, 0, 10],
                           [1, 1, 0]])

loss_matrix_100 = np.array([[0, 1, 100],
                            [1, 0, 100],
                            [1, 1, 0]])

# set loss matrix here
loss_matrix = loss_matrix_100

# conditional pdfs, each class 2 std dev away from each other
pY1  = 0.3
mu1 = np.array([0, 0, 0])
sigma1 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])

pY2 = 0.3
mu2 = np.array([4, 0, 0])
sigma2 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])

pY3 = 0.3
mu3 = 0.5 * np.array([2, 3.464, 0]) + 0.5 * np.array([-2, 3.464, 0])
sigma3 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])

class_priors = [pY1, pY2, pY3]

# generate 10k samples from this setup, keep track of true-hypothesis (true-class) labels
labels = []
samples = []
numSamples = 10000
numLabel1 = 0
numLabel2 = 0
numLabel3 = 0
for i in range(numSamples):
    r = rnd.uniform(0, 1)
    if (r < 0.3):
        labels.append(1)
        samples.append(np.random.multivariate_normal(mu1, sigma1))
        numLabel1 += 1
    elif (r > 0.3 and r < 0.6):
        labels.append(2)
        samples.append(np.random.multivariate_normal(mu2, sigma2))
        numLabel2 += 1
    else:
        labels.append(3)
        samples.append(np.random.multivariate_normal(mu3, sigma3))
        numLabel3 += 3


# multi-dimen gaussian behavior
# types of vectors its more likely, medium likely, less likely to produce
def fXgivenY(x, y):
    if (y == 1):
        mu = mu1
        sigma = sigma1
        py = pY1
    elif (y == 2):
        mu = mu2
        sigma = sigma2
        py = pY2
    elif (y == 3):
        mu = mu3
        sigma = sigma3
        py = pY3
    return np.exp( (-0.5*(x - mu).T) @ np.linalg.inv(sigma) @ (x - mu) ) \
        / ( ((2*np.pi)**(len(x)/2)) * np.sqrt(np.linalg.det(sigma)))

def optimalDecisionRuleWithLoss(x):

    def likelihood_ratio(j):
        return fXgivenY(x, j+1) / fXgivenY(x, 1)

    discriminants = []
    for i in range(num_class):
        discrim = 0
        for j in range(num_class):
            discrim += loss_matrix[i][j] * class_priors[j] * likelihood_ratio(j)
        discriminants.append(discrim)

    return np.argmin(discriminants)+1

conf_matrix = np.zeros((num_class, num_class), float)
def fillConfusionMatrix(d, l):
    if (l == 1):
        numLabel = numLabel1
    if (l == 2):
        numLabel = numLabel2
    else:
        numLabel = numLabel3
    conf_matrix[d-1][l-1] += 1 / numLabel

decisions = []
for i in range(numSamples):
    dec = optimalDecisionRuleWithLoss(samples[i])
    decisions.append(dec)
    fillConfusionMatrix(dec, labels[i])

# for i in range(numSamples):
#     print(
#         "sample class: " + str(labels[i]) + "\t" +
#         "decision made: " + str(decisions[i])
#     )

print("confusion matrix: \n" + str(conf_matrix))

# plot samples, true-lables, and inference success
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(numSamples):

    if labels[i] == decisions[i]:
        c = 'g'
    else:
        c = 'r'

    if labels[i] == 1:
        marker = 'o'
    elif labels[i] == 2:
        marker = '^'
    else:
        marker = 's'

    ax.scatter(samples[i][0], samples[i][1], samples[i][2], c=c, marker=marker, s=100)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()
