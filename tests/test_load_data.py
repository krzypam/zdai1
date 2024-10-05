import sys
sys.path.append('src')

from load_data import load_data

def test_load_data():
  X_train, X_test, y_train, y_test = load_data()
  assert len(X_train) > 100, "train data is too small"