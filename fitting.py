 from sklearn.pipeline import make_pipeline
 clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)) 