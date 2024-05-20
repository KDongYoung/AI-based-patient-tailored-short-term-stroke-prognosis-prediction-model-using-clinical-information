
def Normalize(name, X,Y):
    if name=="standard":
        from sklearn.preprocessing import StandardScaler
        norm=StandardScaler()
        X_norm=norm.fit_transform(X)
    elif name=="minmax":
        from sklearn.preprocessing import MinMaxScaler
        norm=MinMaxScaler()
        X_norm=norm.fit_transform(X)
    elif name=="robust":
        from sklearn.preprocessing import RobustScaler
        norm=RobustScaler()
        X_norm=norm.fit_transform(X)
    else:
        X_norm = X.to_numpy()
    
    return X_norm, Y


