#coding:utf-8
"""
@auther tozawa
@date 2018-8-7
* I just copy the following code
* https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm
* References
* https://distill.pub/2016/misread-tsne/
* https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/
"""
# That's an impressive list of imports
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist # ??

# We import sklearn
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities, _kl_divergence)
#from sklearn.utils.extmath import _ravel

# Random state
RS = 20150101

# We'll use matplotlob for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context('notebook', font_scale=1.5,
                    rc={'lines.linewidth':2.5})

# We'll generate an animation with matplotlib and moviepy
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

"""
* Visualizing handwritten digits
"""
digits = load_digits()
print(digits.data.shape) # (1797, 64) = (number-of-data, dimentions)
print(digits['DESCR'])

nrows, ncols = 2, 5
plt.figure(figsize=(6,3))
plt.gray()
for i in range(ncols*nrows):
    ax = plt.subplot(nrows, ncols, i+1)
    ax.matshow(digits.images[i,...])
    plt.xticks([])
    plt.yticks([])
    plt.title(digits.target[i])
plt.savefig('images/digits-generated.png', dpi=150)

# We first reorder the data points according to the handwritten numbers
X = np.vstack([digits.data[digits.target==i] for i in range(10)]) # (1797, 64)
y = np.hstack([digits.target[digits.target==i] for i in range(10)]) # (1797, )

digits_proj = TSNE(random_state=RS).fit_transform(X)

def scatter(x, colors):
    # We choose a color palette with seaborn
    palette = np.array(sns.color_palette('hls', 10))

    # We create a scatter plot
    f = plt.figure(figsize=(8,8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                        c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digits
    txts = []
    for i in range(10):
        # Position of each labels
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

scatter(digits_proj, y)
plt.savefig('images/digits_tsne-generated.png', dpi=120)

"""
* Similarity matrix
"""
def _joint_probabilities_constant_sigma(D, sigma):
    P = np.exp(-D**2/(2*sigma**2))
    P /= np.sum(P, axis=1)
    return P

# Pairwise distances between all data points
D = pairwise_distances(X, squared=True) # (1797, 1797) The component of diag. is 0.
# Similarity with constant sigma.
P_constant = _joint_probabilities_constant_sigma(D, .002)
# Similarity with variable sigma.
P_binary = _joint_probabilities(D, 30., False) # ???
# The output of this function needs to be reshaped to a square matrix.
P_binary_s = squareform(P_binary)

# We can now display the distance matrix of the data points
plt.figure(figsize=(12,4))
pal = sns.light_palette('blue', as_cmap=True)

plt.subplot(131)
plt.imshow(D[::10, ::10], interpolation='none', cmap=pal)
plt.axis('off')
plt.title('Distance matrix', fontdict={'fontsize': 16})

plt.subplot(132)
plt.imshow(P_constant[::10, ::10], interpolation='none', cmap=pal)
plt.axis('off')
plt.title(r'$p_{j|i}$ (constant $\sigma$)', fontdict={'fontsize': 16})

plt.subplot(133)
plt.imshow(P_binary_s[::10, ::10], interpolation='none', cmap=pal)
plt.axis('off')
plt.title(r'$p_{j|i}$ (variable $\sigma$)', fontdict={'fontsize': 16})
plt.savefig('images/similarity-generated.png', dpi=120)

# This list will contain the positions of the map points at every interation.
positions = []
def _gradient_descent(objective, p0, it, n_iter,
                      n_iter_check=1, n_iter_without_progress=30,
                      momentum=0.5, learning_rate=1000.0, min_gain=0.01,
                      min_grad_norm=1e-7,  min_error_diff=1e-7, verbose=0, args=None, kwargs=None):
    # The documentation of this function can be found in scikit-learn's code.
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_iter = 0

    for i in range(it, n_iter):
        # We save the current position.
        positions.append(p.copy())

        new_error, grad = objective(p, *args, **kwargs)
        error_diff = np.abs(new_error - error)
        error = new_error
        grad_norm = linalg.norm(grad)

        if error < best_iter:
            best_iter = error
            best_iter = i
        elif i - best_iter > n_iter_without_progress:
            break
        if min_grad_norm >= grad_norm:
            break
        if min_error_diff >= error_diff:
            break

        inc = update * grad >= 0.0
        dec = np.invert(inc)
        gains[inc] += 0.05
        gains[dec] *= 0.95
        np.clip(gains, min_gain, np.inf)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

    return p, error, i

sklearn.manifold.t_sne._gradient_descent = _gradient_descent

# Let's run the argorithm again
X_proj = TSNE(random_state = RS).fit_transform(X)

X_iter = np.dstack(position.reshape(-1, 2) for position in positions)

f, ax, sc, txts = scatter(X_iter[..., -1], y)

def make_frame_mpl(t):
    i = int(t*40)
    x = X_iter[..., i]
    sc.set_offsets(x)
    for j, txt in zip(range(10), txts):
        xtext, ytext = np.median(x[y == j, :], axis=0)
        txt.set_x(xtext)
        txt.set_y(ytext)
    return mplfig_to_npimage(f)

animation = mpy.VideoClip(make_frame_mpl, duration=X_iter.shape[2]/40.)
animation.write_gif("images/animation-94a2c1ff.gif", fps=20)

"""
* Let's also create an animation of the similarity matrix of the map points
"""
n = 1. / (pdist(X_iter[..., -1], 'sqeuclidean') + 1)
Q = n / (2.0 * np.sum(n))
Q = squareform(Q)

f = plt.figure(figsize=(6,6))
ax = plt.subplot(aspect='equal')
im = ax.imshow(Q, interpolation='none', cmap=pal)
plt.axis('tight')
plt.axis('off')

def make_frame_mpl_for_map_points(t):
    i = int(t*40)
    n = 1. / (pdist(X_iter[..., i], 'sqeuclidean')+1)
    Q = n / (2.0 * np.sum(n))
    Q = squareform(Q)
    im.set_data(Q)
    return mplfig_to_npimage(f)

animation = mpy.VideoClip(make_frame_mpl_for_map_points,
                            duration=X_iter.shape[2]/40.)
animation.write_gif('images/animation_matrix-da2d5f1b.gif', fps=20)

"""
* Show the distribution of the distances of these points, for different dimensions
"""
npoints = 1000
plt.figure(figsize=(15,4))
for i, D in enumerate((2, 5, 10)):
    # Normally distributed points
    u = np.random.randn(npoints, D)
    # Now on the sphere
    u /= norm(u, axis=1)[:, None]
    # Uniform radius
    r = np.random.rand(npoints, 1)
    # Uniformly within the ball
    points = u * r**(1./D)
    # Plot
    ax = plt.subplot(1, 3, i+1)
    ax.set_xlabel('Ball radius')
    if i == 0:
        ax.set_ylabel('Distance from origin')
    ax.hist(norm(points, axis=1),
            bins=np.linspace(0., 1., 50))
    ax.set_title('D=%d' %D, loc='left')

plt.savefig('images/spheres-generated.png', dpi=100, bbox_inches='tight')

z = np.linspace(0., 5., 1000)
gauss = np.exp(-z**2)
cauchy = 1/(1+z**2)
plt.figure()
plt.plot(z, gauss, label='Gaussian distribution')
plt.plot(z, cauchy, label='Cauchy distribution')
plt.legend()
plt.savefig('images/distributions-generated.png', dpi=100)
