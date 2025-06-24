import pytest
import pandas as pd
from train import read_data, split_data, train_model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Global variable to reuse in tests
sample_data = None

def read_and_assert_data(data_path):
    global sample_data
    df = read_data(data_path)
    sample_data = df
    assert df.shape[1] == 10

def test_read_data():
    assert sample_data is not None
    assert sample_data.shape[1] == 10

def test_train_model():
    assert sample_data is not None
    X = sample_data.drop(columns=["Potability"])
    y = sample_data["Potability"]

    pipe = train_model(X, X, y, y)
    assert isinstance(pipe, Pipeline)
    assert isinstance(pipe.named_steps["Scale"], StandardScaler)
    assert isinstance(pipe.named_steps["Train"], RandomForestClassifier)

if __name__ == "__main__":
    data_path = r"D:\Azure MLOps learning\Custom project\water quality\data_testing\testingdata.csv"
    read_and_assert_data(data_path)
    test_read_data()
    test_train_model()
