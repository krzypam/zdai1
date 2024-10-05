from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from joblib import dump
import mlflow
import sys

sys.path.append('src')
from load_data import load_data

def train():
  with mlflow.start_run():
    X_train, X_test, y_train, y_test = load_data()
    model = LinearRegression()
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(mae)
    dump(model, 'model/model.joblib')
    mlflow.log_metric("mae", mae)
    return mae

if __name__ == '__main__':
  train()