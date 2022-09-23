from config.config import config
from pathlib import Path
import pandas as pd
import os


class DataReader:
    def __init__(self, config:dict):
        self.root = config['root_dir']
        self.asq_path_raw = self.root.joinpath(config['asq_path_raw'])
        self.read_raw_asq()

    def read_raw_asq(self):
        """Read the raw ASQ"""
        if self.asq_path_raw.is_file():
            print(f'\nReading raw ASQ')
            self.raw_asq = pd.read_csv(filepath_or_buffer=self.asq_path_raw,low_memory=False)
            self.raw_asq.drop(labels='Unnamed: 0', axis=1, inplace=True)

    def get_raw_asq(self):
        print(f'\nGet method ASQ dimensions: {self.raw_asq.shape}')
        return self.raw_asq




# %% main to data reader
if __name__ == '__main__':
    reader = DataReader(config=config)

