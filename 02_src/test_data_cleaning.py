import unittest
import pandas as pd
from data_cleaning import clean_data

class TestDataCleaning(unittest.TestCase):
    def test_clean_data_removes_negatives(self):
        df = pd.DataFrame({'calls': [10, -5, 0]})
        cleaned = clean_data(df)
        self.assertTrue((cleaned['calls'] >= 0).all())

if __name__ == '__main__':
    unittest.main()
