import pandas as pd

def Feature_Selection(name, dataset, target, seed):
    print("Feature selection", end=' ') 
    X,Y= dataset.loc[:, dataset.columns != target], dataset.loc[:, target]
    
    if name=="rfec": 
        from sklearn.feature_selection import RFECV
        from xgboost import XGBClassifier
        xgb=XGBClassifier(n_estimators=100, random_state=seed)
        col=X.columns
        
        # RFE+CV(Cross Validation), 10개의 폴드, 5개씩 제거
        rfe_cv = RFECV(estimator=xgb, step=5, cv=10, scoring="f1_macro") 
        rfe_cv.fit(X, Y)
        
        X = rfe_cv.transform(X) # rank가 1인 피쳐들만 선택, rfe_cv.ranking_=1
        feature=[col[i] for i in rfe_cv.get_support(indices=True)]
        print(F"{len(feature)} feature selected by RFECV: {feature}") # 선택된 피쳐들의 이름
    
    else:
        pass

    return pd.concat([pd.DataFrame(X), Y.reset_index(drop=True)], axis=1)     
    
