from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def build_and_train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model, X_test, y_test
