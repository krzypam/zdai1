import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
  df = pd.read_csv('data/otodom.csv')
  df = df[df['price'].notna()]
  X = df[['area']]
  y = df['price']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  return X_train, X_test, y_train, y_test

if __name__ == '__main__':
  load_data()