"""
The Dictionary_Reader reader class read the complete asq information, where the questions and answer options are
written. To be able to user an automatic detection and identification of the questions types, the class
Dictionary_Reader read the table and store the asq information in a dictionary format. Where, the keys are the
questions codes and the values the different possible answers the patient can response.

Useful for the other classes identify the 0 type of question and modify the missing values.

"""

from typing import Optional
from config import config
import numpy as np
import pandas as pd
import pickle
import pathlib
import re


class Dictionary_Reader():
    """
    The selected questions from the asq_dictionary will be used to extract the desired columns from the asq and
    apply the appropiate preprocessing
    """

    def __init__(self, config: dict):
        self.config = config

        if not self.check_dictionary_pickle():
            self.read_asq_dictionary()
            self.construct_dictionary()
            self.save_dictionary_pickle()

    def read_asq_dictionary(self):
        """The ASQ dictionary has the code of the questions, their explanation and the branching logic they have"""
        root = pathlib.Path(__file__).parents[1]
        if 'mnt' in str(root):
            # linux path
            file = root.joinpath(self.config['asq_dictionary'].replace('\\', '/'))
        else:
            file = root.joinpath(self.config['asq_dictionary'])
        if file.is_file():
            if 'asq_dictionary' in self.config.keys():
                self.dict_asq = pd.read_excel(io=file, sheet_name='ASHQ Data Dictionary',
                                              header=0, )
                self.dict_asq.drop(labels=self.config['asq_dict_drop_columns'], axis=1, inplace=True)
                self.dict_asq = self.dict_asq[self.config['asq_dict_columns_interest']]

                self.drop_unwanted_rows()
                self.demo_rows()

            else:
                raise ValueError('asq_dictionary not found in config dictionary \n')
        else:
            raise ValueError(f'Dictionary file not found in {file} \n')

    def drop_unwanted_rows(self):
        """
        General rows that want to be removed, the ramification to questions with 'Description' are not of
        interested in addition to the questions regarding an ID.
        The method stores in a dictionary the 'removed_rows' the rows that are removed from the dictionary so we can
        keep track of the modifications.
        """
        REMOV_ROWS = [*range(0, 72)]
        REMOV_ROWS.extend([75, 76, 89, 90, 91, 92, 93, 94, 95, 97])
        REMOV_ROWS.extend([100, 103, 104, 105, 106, 108, 120, 121, 123, 124])
        DISCARD_STR = ['Other Description', 'Other Details', 'Description', 'List']

        if self.dict_asq['Table_Field Root Name'][71] == 'dem_survey_id':
            self.dict_asq.drop(labels=REMOV_ROWS, axis=0, inplace=True)
            print(f'Removing rows : \n ')
            self.dict_asq.reset_index(inplace=True, drop=True)

            print(f'Removed rows : \n ')
            for rows in REMOV_ROWS:
                print(self.dict_asq['Table_Field Root Name'][rows])
            print('\n')

            self.removed_rows = {}
            for tab_field_name_row, q_name_row in zip([*self.dict_asq['Table_Field Root Name']],
                                                      [*self.dict_asq['Question Name (Abbreviated)']]):
                for dscd in DISCARD_STR:
                    if dscd in q_name_row or 'ID' in q_name_row:
                        self.removed_rows[tab_field_name_row] = q_name_row
                        self.dict_asq = self.dict_asq[self.dict_asq['Table_Field Root Name'] != tab_field_name_row]
                self.dict_asq.reset_index(inplace=True, drop=True)
        else:
            raise ValueError(f' From the ASQ Dictionary, the row "irb_start"" is not number {71}  \n')

    def demo_rows(self):
        """
        Branching questions from demographics that might not be useful, hence we drop them.
        dem_0900 -> Hispanic Origin
            if True, its ramifications details more the origin . We do not need that info
        dem_1000 -> Racial Category - Main
            if 1 or 4 or 5,
                then go to DEM_1010 (Racial Category - Sub)
            If 2,
                SKIP
            If 3 or 6,
                SKIP to DEM_1020 (Racial Category - Country/Province(s) of Origin Details)

        :return: 
        """
        dem_rows_lst = ['dem_0910', 'dem_1000', 'dem_1010', 'dem_1020']
        dem_rows = {}
        for tab_field_name_row, q_name_row in zip([*self.dict_asq['Table_Field Root Name']],
                                                  [*self.dict_asq['Question Name (Abbreviated)']]):
            if tab_field_name_row in dem_rows_lst:
                dem_rows[tab_field_name_row] = q_name_row
                self.dict_asq = self.dict_asq[self.dict_asq['Table_Field Root Name'] != tab_field_name_row]
        self.dict_asq.reset_index(inplace=True, drop=True)
        if dem_rows:
            print(f'Removed rows : \n {dem_rows} ')
            self.removed_rows.update(dem_rows)
        else:
            raise ValueError(f'\nUnable to remove rows: {dem_rows_lst}')

    def construct_dictionary(self) -> dict:
        """
        for row in Numeric scoring Code:
            split(',', str)
            for splits in code:
                if Other in splits
                    add this number to a dictionary as is an answer number we will not be using

        We only take the questions with numeric scoring. Consequently, when the option 'Other' is presented we must
        decide what to do in such cases. Options:
            1. Drop the patient (not recommended)
            2. Set feature as if they do not have a complaint, and we remain in a more narrow domain
            3.
        :return:
        """

        # self.dict_asq = self.dict_asq[self.config['asq_dict_columns_interest']]
        dict_asq = {}
        for row in range(0, self.dict_asq.shape[0]):
            score_code = self.dict_asq['Numeric Scoring Code '][row]
            root_name = self.dict_asq['Table_Field Root Name'][row]
            print(row)
            if root_name == 'mdhx_1920':
                print('debug')
            root_name_category = root_name.split('_')[0].upper()
            # phptab_name = self.dict_asq['PHPTable Name'][row]
            # if phptab_name == 'SCHED':
            #     self._construct_dictionary_sched(row)
            #
            # def _construct_dictionary_sched(self, row) -> dict:
            #     score_code = self.dict_asq['Numeric Scoring Code '][row]
            #     root_name = self.dict_asq['Table_Field Root Name'][row]
            #
            #     if self.dict_asq['Units'][row] == 'identifier' or self.dict_asq['Field Type'][row] == 'timestamp':
            #         return {}
            #
            #     if isinstance(score_code, str):
            if root_name_category == 'PSF' or root_name == 'pap_1500':
                # personal satisfactory questionnaire for answering the ASQ. Not needed
                continue

            if isinstance(score_code, str):
                # and self.dict_asq['Field Type'][row] != 'varchar': (list types are varchar -_-)
                dict_asq[root_name] = {}
                # varchar are related to open questions
                if 'psf' in root_name.lower():
                    continue
                score_code = re.sub(r'\([^)]*\)', '', score_code)  # remove text within parenthesis
                if '\n' in score_code:
                    score_code = score_code.replace('\n', ',')
                if root_name == 'score':
                    score_code = score_code.split(';')
                    for code in score_code:
                        dict_asq[root_name][code.split('=')[0].replace(' ', '')] = \
                            code.split('=')[1]
                    continue

                if root_name == 'cir_0700':
                    score_code = score_code.split(';')
                    score_code[0] = score_code[0].split(',')[1]
                    for code in score_code[1:]:
                        dict_asq[root_name][code.split('=')[0].replace(' ', '')] = \
                            code.split('=')[1]
                    continue

                if root_name == 'narc_1600' or root_name == 'narc_1610' or root_name == 'narc_1620':
                    dict_asq[root_name][1] = 'frequency_number_count'
                    continue

                if root_name == 'narc_1900' or root_name == 'narc_2000' or root_name == 'narc_2100' \
                        or root_name == 'narc_2200' or root_name == 'narc_2110':
                    dict_asq[root_name] = score_code.replace(' ', '_')
                    continue

                if root_name == 'par_0100' or root_name == 'par_0300':
                    score_code = score_code.split(',')
                    del score_code[0]  # remove 'If Never checked'
                    dict_asq[root_name][0] = 'Never'
                    if root_name == 'par_0100': dict_asq[root_name][1] = 'frequency_number_count'
                    if root_name == 'par_0300': dict_asq[root_name][1] = 'frequency_number_count_old'
                    dict_asq[root_name][-55] = 'Don\'tKnow'
                    continue

                if root_name == 'rls_0800' or root_name == 'rls_0801' or root_name == 'narc_1700' \
                        or root_name == 'par_0230' or root_name == 'par_0530':
                    # how olr where you with do not know option
                    dict_asq[root_name][1] = 'frequency_number_count_old'
                    dict_asq[root_name][-55] = 'Don\'tKnow'

                if root_name == 'par_0101':
                    dict_asq[root_name][0] = 'No'
                    dict_asq[root_name][1] = 'Yes'
                    dict_asq[root_name][-1] = 'Never'  # if Never PAR_0100=0 and PAR__0110=1,
                    dict_asq[root_name][-55] = 'Don\'tKnow'
                    continue

                if root_name == 'par_0230':
                    dict_asq[root_name][1] = 'frequency_number_count_age'
                    dict_asq[root_name][-55] = 'Don\'tKnow'
                    continue

                # if root_name == 'par_0500':
                #     dict_asq[root_name][0] = 'Never'
                #     dict_asq[root_name][1] = 'frequency_number_count'
                #     dict_asq[root_name][-55] = 'Don\'tKnow'
                #     continue

                if root_name == 'par_0520':
                    dict_asq[root_name][0] = 'Never'
                    dict_asq[root_name][1] = 'frequency_number_count'
                    dict_asq[root_name][-55] = 'Don\'tKnow'
                    continue

                score_code = score_code.split(',')

                if root_name == 'rMEQ Total Score':
                    continue

                for code in score_code:

                    if not isinstance(code, str):
                        dict_asq[root_name] = np.nan

                    # Run a debug in this if to check all the inputs that are modified by it
                    if code.lower() == 'if never or rarely checked' or code.lower() == 'if never checked':
                        # for cases as = if Never checked, PAR_0500=0 and _0510=1, -55=Don't Know
                        if 'number' in self.dict_asq['Question Name (Abbreviated)'][row].lower():
                            dict_asq[root_name] = 'frequency_number_count'
                        else:
                            dict_asq[root_name] = np.nan
                        break

                    if code.lower() == ' if yes':
                        # When numeric scoring is presented as:
                        # 0=No, 1=Yes, if Yes, PAR_0300=0 and _0310=1, -55=Don't Know
                        #  we want to break the code once the 'if yes' has been reached
                        break

                    if 'if never' in code.lower() or 'if dk' in code.lower():
                        # at this point code has if Never PAR_0200=0 and _0210=1 or If DK, PAR_0200 = -55 = Don't Know
                        if "-55=Don't Know" in score_code[-1]:
                            dict_asq[root_name][-55] = "Don'tKnow"
                        # code.split(root_name_category)[0].replace(' ', '')
                        break

                    if '=' in code:
                        if 'Range' in code:
                            # not a value with information
                            continue
                        if '-' in code or ':' in code:
                            if '-' in code.split('=')[0].replace(' ', '') and \
                                    code.split('=')[0].replace(' ', '') != '-55' and \
                                    code.split('=')[0].replace(' ', '') != '-44':
                                # e.g. phq_1000 -> '5-9=Minimal Symptoms' and avoid the -55 and -44
                                dict_asq[root_name][code.split('=')[0].replace(' ', '')] = \
                                    code.split('=')[1].replace(' ', '')
                                continue
                            else:
                                dict_asq[root_name][int(code.split('=')[0].replace(' ', ''))] = \
                                    code.split('=')[1].replace(' ', '')
                            continue

                        if '<' in code or '≥' in code or '>' in code:
                            dict_asq[root_name][code.split('=')[0]] = code.split('=')[1].replace(' ', '')
                            continue

                        else:
                            dict_asq[root_name][int(code.split('=')[0])] = code.split('=')[1].replace(' ', '')
                            continue
                    else:
                        continue

            # elif root_name_category not in self.dict_asq['Question Name (Abbreviated)'][row].lower() and \
            #         (self.dict_asq['Units'][row] != 'identifier' or self.dict_asq['Field Type'][row] != 'timestamp'):
            elif self.dict_asq['Units'][row] == 'identifier' or self.dict_asq['Field Type'][row] == 'timestamp' \
                    or self.dict_asq['Field Type'][row] == 'varchar':
                # or     pd.isna(self.dict_asq['Units'][row]):
                # narc_1800
                print(f'\nIgnored question: {root_name}')
                continue

            else:
                # elif self.dict_asq['Units'][row] != 'identifier' or self.dict_asq['Field Type'][row] != 'timestamp':
                # numeric_scoring_code is N/A, happens for continues variables. Although, timestamp and identifier
                # also have N/A in numeric_scoring_code. So we filter them
                dict_asq[root_name] = {}
                if 'per week' in self.dict_asq['Question Name (Abbreviated)'][row].lower():
                    dict_asq[root_name] = 'frequency_number_count_per_week'

                elif 'time' in self.dict_asq['Question Name (Abbreviated)'][row].lower():
                    if 'hours' in self.dict_asq['Question Name (Abbreviated)'][row].lower():
                        dict_asq[root_name] = 'time_hours'
                        continue
                    if 'minutes' in self.dict_asq['Question Name (Abbreviated)'][row].lower():
                        dict_asq[root_name] = 'time_minutes'
                        continue
                    if 'number' in self.dict_asq['Question Name (Abbreviated)'][row].lower():
                        # Exercise - Number of times
                        dict_asq[root_name] = 'frequency_number_count_times'
                        continue
                    else:
                        dict_asq[root_name] = 'time'
                        continue

                elif 'number' in self.dict_asq['Question Name (Abbreviated)'][row].lower():
                    # dem_0100
                    dict_asq[root_name] = 'frequency_number_count'
                    continue

                if root_name_category.lower() == 'dem':
                    dict_asq[root_name] = self.dict_asq['Question Name (Abbreviated)'][row].lower().replace('-', '') \
                        .replace(' ', '_')
                    continue

                if root_name_category.lower() == 'sched':
                    dict_asq[root_name] = self.dict_asq['Question Name (Abbreviated)'][row].lower().replace('-', '') \
                        .replace(' ', '_')
                    continue

                else:
                    if 'please approximate the percent of 24 hour food intake for each meal ' in \
                            self.dict_asq['Question Name (Abbreviated)'][row].lower().split('-')[0]:
                        meal_type = self.dict_asq['Question Name (Abbreviated)'][
                            row].lower().split('-')[1].replace(' ', '')
                        dict_asq[root_name]['perc_24h_food_intake_for_' + meal_type[0] + meal_type[-1]] = meal_type
                        continue
                    else:
                        # e.g,narc_1900
                        dict_asq[root_name] = self.dict_asq['Question Name (Abbreviated)'][row].lower()
                        continue

        dict_asq['FOSQ Score'] = 'formula'
        del dict_asq['isq_1400']
        self.dictionary_questions = dict_asq
        if hasattr(self, 'dictionary_questions'):
            print('\n self.dictionary_questions create (keys are the questions to consider in the model)')
            self.save_dictionary_pickle()
            return self.dictionary_questions
        else:
            raise ValueError('\n Unable to retrieve self.dictionary_questions ')

    def save_dictionary_pickle(self, pre_processed: Optional[bool] = False):
        """
        Save the dictionary as pickle
        :return:
        """
        if not pre_processed:
            if hasattr(self, 'dictionary_questions'):  # is a dictionary
                root = pathlib.Path(__file__).parents[1]
                file = root.joinpath(self.config['asq_dictionary']).parents[0]
                file = file.joinpath('dictionary_questions.pickle')
                with open(file, 'wb') as handle:
                    pickle.dump(self.dictionary_questions, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                raise ValueError('\n Unable to save self.dictionary_questions. Object not found ')
        elif pre_processed:
            if hasattr(self, 'preproc_asq_dict'):  # is a dictionary
                root = pathlib.Path(__file__).parents[1]
                file = root.joinpath(self.config['pre_processed_asq_dictionary'])
                if not file.exists():
                    file.mkdir(parents=True, exist_ok=True)
                file = file.joinpath('preproc_dictionary_questions.pickle')
                with open(file, 'wb') as handle:
                    pickle.dump(self.preproc_asq_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                raise ValueError('\n Unable to save self.preproc_asq_dict. Object not found ')

    def check_dictionary_pickle(self) -> bool:
        """
        If the class has been called and the dictionary is saved, then we just need to extract the dictionary as a
        pickle. Return True if the file has been prperly pickle
        :return:
        """
        root = pathlib.Path(__file__).parents[1]
        file = root.joinpath(self.config['asq_dictionary']).parents[0]
        file = file.joinpath('dictionary_questions.pickle')

        if file.is_file():
            return True
        else:
            print(f'\nUnable to pickle self.dictionary_questions from {file}')
            return False

    def get_asq_dictionary(self, pre_processed: Optional[bool] = False):
        """
        If the class has been called and the dictionary is saved, then we just need to extract the dictionary as a
        pickle. Return True if the file has been properly pickle
        :return:
        """
        root = pathlib.Path(__file__).parents[1]
        if not pre_processed:
            file = root.joinpath(self.config['asq_dictionary']).parents[0]
            file = file.joinpath('dictionary_questions.pickle')

            if file.is_file():
                with open(file, 'rb') as handle:
                    self.asq_dictionary = pickle.load(handle)
                    # pickle.dump(self.dictionary_questions, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print(f'\nUnable to pickle self.dictionary_questions from {file}')

            if hasattr(self, 'asq_dictionary'):
                print(f'\n self.asq_dictionary has been properly pickle from {file} '
                      f'\n Call self.asq_dictionary to use it')
                return self.asq_dictionary

            else:
                print(f'\nUnable to include self.asq_dictionary to the class')
                return False

        if pre_processed:
            file = root.joinpath(self.config['pre_processed_asq_dictionary'])
            file = file.joinpath('preproc_dictionary_questions.pickle')
            with open(file, 'rb') as handle:
                self.preproc_asq_dict = pickle.load(handle)
            if hasattr(self, 'preproc_asq_dict'):
                print(f'\nself.preproc_asq_dict has been properly pickle from {file} '
                      f'\n Call self.preproc_asq_dict to use it')
                return self.preproc_asq_dict

            else:
                print(f'\nUnable to include self.preproc_asq_dict to the class')
                return False


if __name__ == "__main__":
    dataset_info = Dictionary_Reader(config=config)
