import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거


def Balance(balance_name, seed, X,Y):
    
    if balance_name=="ROS": # random_oversampling
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=seed)
        X, Y = sampler.fit_resample(X, Y)
        
    elif balance_name=="CWROS": # weighted_random_oversampling
        from collections import Counter
        counts = Counter()
        classes = [] 

        # calculate class
        for y in Y: 
            counts[int(y)] += 1 
            classes.append(y) 
        n_classes = len(counts)

        # calculate weight
        weight_per_class = {}
        for y in counts: 
            weight_per_class[y] = 1 / (counts[y] * n_classes)
        weights = np.zeros(len(Y))
        for i, y in enumerate(classes):
            weights[i] = weight_per_class[int(y)]

        return X, Y, weights
    
    elif balance_name=="SMOTE":
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=seed)
        X, Y = smote.fit_resample(X, Y)
                
    elif balance_name=="ADASYN":
        from imblearn.over_sampling import ADASYN
        adasyn = ADASYN(random_state=seed)
        X, Y = adasyn.fit_resample(X, Y)
        
    else:
        pass
    
    return X, Y, None