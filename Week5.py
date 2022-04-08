
# Number of males/females in our dataset
m_n = 10000
f_n = 500

# Store performance
tr_acc, te_acc, m_te_acc, f_te_acc = [], [], [], []


# Train a couple times (synthetic data performance can vary)
num_iter = 10

for ii in tqdm(range(num_iter)):

    # Generate synthetic data
    dat = fx.Create_MDD(m_n, f_n)
    X, y = dat.X, dat.y

    # Create splits
    kf = KFold(n_splits=5)

    # Train/eval for each fold
    for tr_idx, te_idx in kf.split(y):

        # Split train/test
        tr_x, te_x =  X.iloc[tr_idx, :], X.iloc[te_idx, :]
        tr_y, te_y =  y.iloc[tr_idx], y.iloc[te_idx]

        # Fit our model
        logl1 = LogisticRegression(penalty='l1', solver='liblinear')
        logl1.fit(tr_x, tr_y)

        # Get our predictions
        tr_pred = logl1.predict(tr_x)
        te_pred = logl1.predict(te_x)

        # Get accuracy
        tr_acc.append(accuracy_score(tr_y, tr_pred))
        te_acc.append(accuracy_score(te_y, te_pred))


        '''#################################################
        # Split test into M/F only and evaluate performance
        #################################################'''

        # Stratify test set by sex
        m_te_x = te_x[te_x['sex'] == 0]
        m_te_y = te_y[te_x['sex'] == 0]
        f_te_x = te_x[te_x['sex'] == 1]
        f_te_y = te_y[te_x['sex'] == 1]

        # Get our predictions
        te_pred_m = logl1.predict(m_te_x)
        te_pred_f = logl1.predict(f_te_x)

        # Get accuracy
        m_te_acc.append(accuracy_score(m_te_y, te_pred_m))
        f_te_acc.append(accuracy_score(f_te_y, te_pred_f))


print(f'Train Acc: {np.mean(np.array(tr_acc)):.3f} '
      f'Test Acc: {np.mean(np.array(te_acc)):.3f} '
      f'Test Acc Male: {np.mean(np.array(m_te_acc)):.3f} '
      f'Test Acc Female: {np.mean(np.array(f_te_acc)):.3f}')



# Create the explainer
explainer = shap.LinearExplainer(logl1, tr_x)
shap_values = explainer(tr_x)

# Plot
shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)

# Create the explainer
explainer = shap.LinearExplainer(logl1, tr_x)
shap_values = explainer(tr_x)

# Plot
shap.plots.bar(shap_values.abs.mean(0))
shap.plots.beeswarm(shap_values)
