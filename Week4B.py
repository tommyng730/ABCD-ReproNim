
a, b = 40, 100

mu = (a + b)/2
std = np.sqrt((b-a)**2/12)

print(f'Mean: {mu:.4f} Std: {std:.4f}')


x = []
for ii in range(1000):
    x.append(a + (b -a) * random.random())

mu = sum(x)/len(x)

std = np.sqrt(np.sum(np.subtract(x, mu)**2)/len(x))

print(f'Mean: {mu:.4f} Std: {std:.4f}')
plt.hist(x)
plt.show()


from scipy.stats import norm

mu = 0
std = 1

# Create the distribution
dist = norm(mu, std)

# Range over which we access the pdf/cdf
x = np.linspace(-5, 5, 500)

# Obtain the pdf over this range of x
pdf = dist.pdf(x)

# Obtain the cdf over this range of x
cdf = dist.cdf(x)

# Answers to the problems
prob1 = dist.pdf(.003)
prob2 = dist.cdf(-2)
prob3 = 1 - dist.cdf(1.25)

print(f'Prob 1: {prob1:.3f} Prob 2: {prob2:.3f} Prob 3: {prob3:.3f}')

# Plot this
fig = plt.figure(figsize = (10, 8))
plt.plot(x, pdf, color = 'green', alpha = .4, linewidth = 5, label = 'PDF')
plt.plot(x, cdf, color = 'green', alpha = .4, linewidth = 2, linestyle = '--', label = 'CDF')
plt.vlines(x = .003, ymin = 0, ymax = 1, linewidth = 3, linestyle = '--', color = 'k', label = 'Prob 1')
plt.vlines(x = -2, ymin = 0, ymax = 1, linewidth = 3, linestyle = '--', color = 'r', label = 'Prob 2')
plt.vlines(x = 1.25, ymin = 0, ymax = 1, linewidth = 3, linestyle = '--', color = 'r', label = 'Prob 3')
plt.legend()
plt.show()


from scipy.stats import norm, multivariate_normal

# Generate our X, Y, pos
X, Y, pos = genMesh(-5, 5, 1000)

# Create our array of mu
mus = np.array([0.0, 1.0])

# Create our covariance matrix for INDEPENDENT gaussian rvs
cov = np.array([[1.0, 0],
                [0, 1.0]])

# Create our multivariate normal
rv = multivariate_normal(mus, cov)
Z = rv.pdf(pos)

# Plot this
plot_multivariate_normal(X, Y, Z)



from scipy.stats import norm, multivariate_normal

# Generate our X, Y, pos
X, Y, pos = genMesh(-5, 5, 1000)

# Create our array of mu
mus = np.array([0.0, 1.0])

# Create our covariance matrix for CORRELATED gaussian rvs
cov = np.array([[1.0, .75],
                [.75, 1.0]])

# Create our multivariate normal
rv = multivariate_normal(mus, cov)
Z = rv.pdf(pos)

# Plot this
plot_multivariate_normal(X, Y, Z)


def get_sens_spec(x, p0, p1, mu1, mu2, std1, std2, plot = True):

    # Create distributions of Y given H0/H1
    py_H0 = norm(mu1, std1)
    py_H1 = norm(mu2, std2)

    # Create the pdfs
    pdf_py_H0 = py_H0.pdf(x) * p0
    pdf_py_H1 = py_H1.pdf(x) * p1

    # Create the cdfs
    cdf_py_H0 = py_H0.cdf(x) * p0
    cdf_py_H1 = py_H1.cdf(x) * p1

    # This fancy number will find us the intersectin of our two PDFS
        # credit: https://stackoverflow.com/questions/28766692/intersection-of-two-graphs-in-python-find-the-x-value
    intersect_ind = np.argwhere(np.diff(np.sign(pdf_py_H0 - pdf_py_H1))).flatten()

    # Get the intersections by ind
    intersect_x = x[intersect_ind][0]
    intersect_y = py_H0.pdf(intersect_x) * p0

    if plot:
        plot_hypothesis(x, p0, p1, pdf_py_H0, pdf_py_H1, cdf_py_H0, cdf_py_H1, intersect_x, intersect_y)

    # Return sensitivity, specificity
    TP = 1 - (py_H1.cdf(intersect_x))
    TN = py_H0.cdf(intersect_x)
    FP = 1 - (py_H0.cdf(intersect_x))
    FN = py_H1.cdf(intersect_x)
    sens = TP/(TP + FN)
    spec = TN/(TN + FP)

    print(f'Sens: {sens:.3f} Spec: {spec:.3f}\n')

    return sens, spec


  # Range of x
  x = np.linspace(-5, 5, 500)

  # The priors of H0/H1
  p0, p1 = .5, .5

  # Mean, std for H0/H1
  mu1, mu2 = 0, 2
  std1, std2 = 1, 1

  sens1, spec1 = get_sens_spec(x, p0, p1, mu1, mu2, std1, std2)


# Range of x
x = np.linspace(-5, 5, 500)

# The priors of H0/H1
p0, p1 = .3, .7

# Mean, std for H0/H1
mu1, mu2 = 0, 2
std1, std2 = 1, 1

sens2, spec2 = get_sens_spec(x, p0, p1, mu1, mu2, std1, std2)



# Range of x
x = np.linspace(-5, 5, 500)

# The priors of H0/H1
p0, p1 = .8, .2

# Mean, std for H0/H1
mu1, mu2 = 0, 2
std1, std2 = 1, 1

sens3, spec3 = get_sens_spec(x, p0, p1, mu1, mu2, std1, std2)



# Range of x
x = np.linspace(-5, 5, 500)

# The priors of H0/H1
p0, p1 = .5, .5

# Mean, std for H0/H1
mu1, mu2 = 0, 3.5
std1, std2 = 1, 1

sens4, spec4 = get_sens_spec(x, p0, p1, mu1, mu2, std1, std2)


fig, ax = plt.subplots(figsize = (6, 4))

# Setup lines
ax.plot(np.arange(0, 1, .01), np.arange(0, 1, .01), linewidth = 2, linestyle = '--', color = 'k')
ax.vlines(x = 1, ymin = 0, ymax = 1, linewidth = 2, linestyle = '--', color = 'k')
ax.hlines(y = 1, xmin = 0, xmax = 1, linewidth = 2, linestyle = '--', color = 'k')

# Plot
ax.scatter(1 - spec1, sens1, s = 150, color = 'red', label = 'Prob 1')
ax.scatter(1 - spec2, sens2, s = 150, color = 'blue', label = 'Prob 2')
ax.scatter(1 - spec3, sens3, s = 150, color = 'orange', label = 'Prob 3')
ax.scatter(1 - spec4, sens4, s = 150, color = 'green', label = 'Prob 4')

# Chhanges
ax.set_title('ROC Curve')
ax.set_xlabel('1 - Specificity')
ax.set_ylabel('Sensitivity')
ax.legend(loc = 'lower right')
plt.show()
