import os.path
import pytest


def test_no_data_zip_file():
    assert os.path.isfile("data.zip") == False


def test_no_checkpoint_file():
    assert os.path.isfile("checkpoint.pth") == False


def test_accuracty_above_70():
    with open("metrics.csv") as f:
        lines = f.readlines()
        line = lines[-1]
        fields = line.split(",")
        assert float(fields[4].strip()) > 70.0


def test_accuracy_cats_above_70():
    with open("metrics.csv") as f:
        lines = f.readlines()
        line = lines[-1]
        fields = line.split(",")
        assert float(fields[5].strip()) > 70.0


def test_accuract_dogs_above_70():
    with open("metrics.csv") as f:
        lines = f.readlines()
        line = lines[-1]
        fields = line.split(",")
        assert float(fields[6].strip()) > 70.0