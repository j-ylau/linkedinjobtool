from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
    print(f"R2 Score: {r2_score(y_test, y_pred)}")
