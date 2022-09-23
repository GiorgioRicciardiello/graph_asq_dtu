"""
Configuration file with the static variables
"""
import pathlib

config = {
    'asq_path_raw': pathlib.Path(r'data/raw/complete_dataset_after_reader.csv'),
    'root_dir' : pathlib.Path(__file__).parents[1],
    'seed': 7182,
    'match_subject_ahi_path' : r'data/pre_processed/match_subject_ahi',
    'nonpredictors' : ['subject', 'ahi'],
    'target': 'ahi',
    'ijv_format_folder':  pathlib.Path(r'data/graph/ijv_format'),

    'train_random_effects_epochs_num': 500,
}
