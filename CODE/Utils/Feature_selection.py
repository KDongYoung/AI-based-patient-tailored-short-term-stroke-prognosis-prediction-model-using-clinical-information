import pandas as pd

def Feature_Selection(name, dataset, target, max_iter, seed):
    
    X,Y= dataset.loc[:, dataset.columns != target], dataset.loc[:, target]
    
    if name=="rfec": 
        from sklearn.feature_selection import RFECV
        from xgboost import XGBClassifier
        xgb=XGBClassifier(n_estimators=max_iter,random_state=seed)
        col=X.columns
        # RFE+CV(Cross Validation), 10 fold, remove 5 each
        rfe_cv = RFECV(estimator=xgb, step=5, cv=10, scoring="f1_macro") 
        rfe_cv.fit(X, Y)
        
        X = rfe_cv.transform(X) # rfe_cv.ranking_=1 feature select
        feature=[col[i] for i in rfe_cv.get_support(indices=True)]
        print(F"{len(feature)} Selected feature by RFECV: {feature}") # Selected feature name

    return pd.concat([pd.DataFrame(X), Y.reset_index(drop=True)], axis=1)     
    
