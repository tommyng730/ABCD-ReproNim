To test the reproducibility of the results across groups of subjects, they split the group of subjects in two and learn ICA maps from each sub-group. They compare the overlap of thresholded maps and reorder one set to match maps by maximum overlap. To quantify the reliability of the patterns identified on the full datasets, they select for each pattern extracted from the full dataset the best matching one in the different subsets computed in the cross-validation procedure using Pearson’s correlation coefficient. Along with the extracted maps, they report the average value of this pattern-reproducibility measure.

from nilearn.decomposition import CanICA
from nilearn.plotting import plot_prob_atlas

# Create the ICA model
canica = CanICA(n_components=20,
                memory="nilearn_cache",
                memory_level=2,
                verbose=10,
                mask_strategy='whole-brain-template',
                random_state=0)

print(f'Fitting the CanICA Model to our {len(func_filenames)} subjects...')

# Fit the model to our data
canica.fit(func_filenames)


canica_components_img = canica.components_img_
plot_prob_atlas(canica_components_img, title='All ICA components')

ward = Parcellations(method='ward', n_parcels=1000, standardize=False, smoothing_fwhm=2.)
kmeans = Parcellations(method='kmeans', n_parcels=50, standardize=True, smoothing_fwhm=10.)

ward.fit(dataset.func)
kmeans.fit(dataset.func)


# Create figure
fig = plt.figure(figsize=(30, 16))

#manifold methods
#LLE = manifold.LocallyLinearEmbedding(method="standard",n_neighbors=n_neighbors,n_components=n_components, eigen_solver="auto")
#LLE = manifold.LocallyLinearEmbedding(method="ltsa",n_neighbors=n_neighbors,n_components=n_components, eigen_solver="auto")
#LLE = manifold.LocallyLinearEmbedding(method="hessian",n_neighbors=n_neighbors,n_components=n_components, eigen_solver="auto")
#LLE = manifold.LocallyLinearEmbedding(method="modified",n_neighbors=n_neighbors,n_components=n_components, eigen_solver="auto")
#LLE = manifold.Isomap(n_neighbors=n_neighbors,n_components=n_components)
#LLE = manifold.MDS(n_components=n_components, max_iter=100, n_init=1)
LLE = manifold.SpectralEmbedding(n_neighbors=n_neighbors,n_components=n_components)
#LLE = manifold.TSNE(n_components=n_components, init="pca", random_state=0)

Y = LLE.fit_transform(X)

ax = fig.add_subplot(2, 5, 2)
ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
ax.axis("tight")
