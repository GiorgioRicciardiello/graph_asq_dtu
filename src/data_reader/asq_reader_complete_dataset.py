"""
This asq_reader is utilized for the complete dataset. Class utilized after running asq_added_entries that is dedicated
to match the AHIs and merge the ASQs
"""
from typing import Optional
from tqdm import tqdm
from config import config
from src.data_reader.dictionary_reader import Dictionary_Reader
import numpy as np
import pandas as pd
import pathlib

import matplotlib.pyplot as plt


class ASQ_Reader_Complete_Dataset(Dictionary_Reader):
    """
    Read the merged ASQ and apply pre processing to the columns e.g. transform the nans, hot encoding to list type
    columns, etc...
    """

    def __init__(self, config: dict, override_existing:Optional[bool]=False, days_threshold:Optional[int]=360):
        super().__init__(config)
        self.config = config
        replace_nan_negative = True
        # 'mdhx_0120' in [*self.complete_dataset.columns] this type of columns should no
        self.root = pathlib.Path(__file__).parents[1]
        self.table_dict_manual_removal_path = r'C:\Documents\msc_thesis_project\data\pre_processed\manual_column_removal\table_dict_intersection_complete_dataset_with_stages.xlsx'

        self.modified_answ_path = \
            self.root.joinpath(pathlib.Path(r'data/pre_processed/manual_column_removal/log_modified_answer_response'))

        self.output_path = self.root.joinpath(pathlib.Path(self.config['pre_processed_merged']) /
                                              self.config['complete_dataset_name'].replace('.csv', '_after_reader.csv'))

        remove_screening_columns = {
            'mdhx_0300': [2, 8]
        }
        self.list_type_questions =[]
        if not self.check_complete_dataset(override=override_existing):
            # read data
            self.open_complete_dataset()
            if days_threshold >= 0:
                self.apply_days_threshold(days_threshold_diff=days_threshold)
            self.open_complete_dataset_dictionary()
            # Get the ASQ handbook (dictionary)
            self.asq_dictionary = Dictionary_Reader.get_asq_dictionary(self, pre_processed=False)

            # make table of the frequency response for each predictor
            self.create_new_dictionary_table()
            # start pre-processing
            # nan dedicated managing functions
            self._remove_rows_all_nans()
            self._remove_pap_patients_and_columns()
            self._set_nan_scores_to_zero()
            self._additional_categorical_ordinal_dimension()

            # data transformations
            self._factorize_str_to_int()
            self._hours_to_minutes(result_type_hours=True)
            self._convert_height_meters_and_pounds_to_kg()
            self._remove_next_shift_questions()

            # self.filter_by_threshold(remove_bool=False)

            # remove from the annotated Excel table
            self.manual_removal_from_table()
            print(f'\nDimension after initial pre-processing: {self.complete_dataset.shape}')
            self._decode_listcol_and_screen_subjects(apply_one_hot=True, report=False,
                                                     remove_screening_columns=remove_screening_columns)

            if replace_nan_negative:
                # self._decode_listcol_and_screen_subjects(apply_one_hot=True, report=False)
                self._missing_data_code_mapper(replace_nan_negative=replace_nan_negative)
                self._dtypes_float_to_int()
                self._decode_time_cyclical(all_minutes=True)

            self._remove_constant_columns()
            self.save_frame()
        else:
            print(f'Existing pre-processed dataset not modified. To obtain call self.run()')

    # %%
    def _remove_constant_columns(self):
        """
        Remove constant rows. NaNs will be ignored as it searched for the count of unique values != 1.
        But a column of only nans will be ignored as nan != nan
        :return:
        """
        constant_columns = self.complete_dataset.apply(pd.Series.nunique) != 1
        constant_columns = [*[*self.complete_dataset.columns][constant_columns]]
        self.complete_dataset = self.complete_dataset.loc[:,self.complete_dataset.apply(pd.Series.nunique) != 1]
        print(f'\nConstant columns removed {len(constant_columns)} \n{constant_columns}')

        with open(self.modified_answ_path/'constant_columns.txt', "w") as output:
            output.write(str(constant_columns))


    # %% pre-processing functions
    def _remove_rows_all_nans(self):
        """
        Rows in the frame that have all nan values. They will be removed
        :return:
        """
        predictors = [col_ for col_ in [*self.complete_dataset.columns] if col_ != 'ahi']
        predictors = [col_ for col_ in predictors if col_ != 'subject']
        selected_rows = self.complete_dataset[predictors][self.complete_dataset[predictors].isnull().all(axis=1)]
        selected_rows = [*selected_rows.index]
        if len(selected_rows) == 1:
            self.complete_dataset.drop(labels=selected_rows, axis=0, inplace=True)
            self.complete_dataset.reset_index(drop=True, inplace=True)
            print(f'\nRemoved row of only nans: {selected_rows}')
        else:
            print(f'\nRemoved row of only nans= ')
            for sel_row_ in selected_rows:
                self.complete_dataset.drop(labels=sel_row_, axis=0, inplace=True)
                self.complete_dataset.reset_index(drop=True, inplace=True)
            print(selected_rows)

    def _factorize_str_to_int(self):
        """
        rls_probability have string categorical, we can or factorize the column or apply one hot encoding

        col_string_responses = ['rls_probability', 'soclhx_0101']
        :return:
        """
        # rls probability
        print(f'\nReplacing string values in column "rls_probability" with')
        self.complete_dataset['rls_probability'] = self.complete_dataset['rls_probability'].str.replace('IN', 'in')
        rls_prob = [*self.complete_dataset['rls_probability'].unique()]
        rls_prob = [rls for rls in rls_prob if rls == rls]  # remove the nan
        codes, _ = pd.factorize(rls_prob)
        mapper_rls = {}
        for code, val in zip(codes, rls_prob):
            mapper_rls[val] = code
        print(f'\n= {mapper_rls}')
        self.complete_dataset['rls_probability'].replace(to_replace=mapper_rls, inplace=True)
        self.complete_dataset['rls_probability'].fillna(value=min([*mapper_rls.values()]), inplace=True)

        # soclhx_0101
        print(f'\nReplacing string values in column "soclhx_0101" with')
        soclhx_0101_mapper = {'True': 1, 'False': 0, '0.0': 0, '1.0': 1}
        print(f'\n= {soclhx_0101_mapper}')
        self.complete_dataset['soclhx_0101'].replace(to_replace=soclhx_0101_mapper, inplace=True)

    def _remove_pap_patients_and_columns(self):
        """
        Patients with a pap answer are removed as they can be nuisance factors
        :return:
        """
        removed_subjects_pap = {}
        rows_to_remove = []
        papcol = [pap for pap in [*self.complete_dataset.columns] if 'pap' in pap]
        for pap_col_ in papcol:
            # pap_col_ = papcol[0]
            # bool array, False means a nan, True means a response
            nan_values = self.complete_dataset[pap_col_] == self.complete_dataset[pap_col_]
            # index of the rows with response
            non_nan_index = np.where(nan_values == True)[0].tolist()
            if len(non_nan_index) != 0:
                # subject that answered the cpap questionnaire
                removed_subjects_pap[pap_col_] =self.complete_dataset['subject'][non_nan_index]
                # append the indexes of the rows to remove
                rows_to_remove.extend(non_nan_index)
        # remove unique indexes (to avoid droping the same row twice and causing an error)
        rows_to_remove = list(set(rows_to_remove))
        self.complete_dataset.drop(labels=rows_to_remove, axis=0, inplace=True)
        # reset indexes as wer removed rows
        self.complete_dataset.reset_index(drop=True, inplace=True)
        # drop the pap columns
        self.complete_dataset.drop(labels=papcol, axis=1, inplace=True)
        print(f'\nAll pap columns where removed= {papcol}')
        removed_subjects_pap = pd.DataFrame(removed_subjects_pap)
        removed_subjects_pap.to_excel(excel_writer=self.modified_answ_path / 'pap_subjects_removed.xlsx')
        print(f'\nSubjects with pap answers removed {len(rows_to_remove)}, log saved in \n  {removed_subjects_pap}')

    def _set_nan_scores_to_zero(self):
        """
        All the questions regarding a  SCORE that have nans values will be equal to zero as they are the same
        :return:
        """
        score_list = []
        modified_scores = {}
        min_score = {'cir_0700': 4, 'fosq_1100': 5}
        for col_name_, question_ in zip(self.dataset_dictionary['Column Name'],
                                        self.dataset_dictionary['Question Name (Abbreviated)']):
            if 'score' in col_name_ or 'score' in question_.lower() or 'lr' in col_name_ or 'probability' in col_name_:
                print(f'\nScore column: {col_name_}')
                score_list.append(col_name_)

        for col_score_ in score_list:
            if col_score_ in [*self.complete_dataset.columns]:
                if self.complete_dataset[col_score_].isnull().values.any():
                    if col_score_ in [min_score.keys()]:
                        # the minimum score is not zero but other constant
                        self.complete_dataset.loc[:, col_score_].fillna(value=min_score[col_score_], inplace=True)
                        modified_scores[col_score_] = {
                            'question': col_score_,
                            'action': f'nan == {min_score[col_score_]}. Nans replaces by zero',
                            'type': 'score'}
                    else:
                        self.complete_dataset.loc[:, col_score_].fillna(value=0, inplace=True)
                        modified_scores[col_score_] = {
                            'question': col_score_,
                            'action': 'nan == 0 . Nans replaced by zero',
                            'type': 'score'}
                else:
                    print(f' Score no missing values: {col_score_}')
        modified_scores = pd.DataFrame(modified_scores).T
        modified_scores.to_excel(excel_writer=self.modified_answ_path / 'nan_to_zero_scores.xlsx')
        print(f'\nScores columns with nan values had been replaces with zeros: {score_list}')

    def _hours_to_minutes(self, result_type_hours: Optional[bool] == True):
        """
        We have redundant questions that ask the same information, one in hours and the other rin minutes. To reduce
        the dimensionality, we will convert all the hour formats to minutes and later merge them with their respective
        minutes similar question.
        keep_hour_format == True will convert the hours_minutes to hours and remove the minutes format column. This
        helps to reduce the dimensionality of the column. It retains a less spare domain.

        Search in the 'Question Name (Abbreviated)' column for the hour string. Convert to minutes, then search for a
        similar quesiton but the minutes substring in it. Then we merge them, remove the question in the hour format and
        remove the minutes substring from the questions.

        The hours and minutes substring is found at the end the column row.

        we need to skip the negative number if not we will have new subjects rows with ID == 0
        :return:
        """
        keep_hour_format = result_type_hours

        if keep_hour_format:
            # the hours columns will be kept and minutes will be removed
            modified_hours = []
            droped_minutes = {}
        else:
            modified_minutes = []
            droped_hours = {}

        for question_ in self.dataset_dictionary['Question Name (Abbreviated)'].items():
            col_name_h = self.dataset_dictionary.loc[question_[0], 'Column Name']
            question_ = question_[1]
            # question_ = 'Sleep Latency Week Night / School Night / Work Night or Day (Current Shift) - Hours'
            # col_name_h = 'sched_2200'

            # question_ = '0=Very good, 1=Good, 2=Fair, 3=Poor, 4= Very poor'
            # col_name_h = 'sched_3100'
            if isinstance(question_, str):
                question_ = question_.lower()
                hour_substring_index = question_.find(' - hours')
                if hour_substring_index > 0:
                    # found an hour substring
                    if col_name_h in [*self.complete_dataset.columns]:
                        # convert hours to minutes
                        col_hour_to_min = self.complete_dataset.loc[:, col_name_h] * 60
                        # replace nan by zero
                        # hours_to_min_nan_idx = np.where(np.asanyarray(np.isnan(col_hour_to_min)))[0].tolist()
                        # col_hour_to_min.fillna(value=0, inplace=True, axis=0)
                        if not keep_hour_format:
                            # remove the hours columns
                            self.complete_dataset.drop(labels=col_name_h, axis=1, inplace=True)
                            droped_hours[col_name_h] = question_
                        # get the respective minutes columns (col_name_+10)
                        col_minutes = col_name_h.split('_')[1]
                        col_minutes = col_minutes[0:-2] + col_minutes[-2::].replace('00', '10')
                        col_minutes = col_name_h.split('_')[0] + '_' + col_minutes
                        if col_minutes in [*self.complete_dataset.columns]:
                            if keep_hour_format:
                                # we  want to keep the missing codes (negative numbers -44, -99, ...)
                                non_neg_index_hours = [*self.complete_dataset.loc[:,
                                                        col_name_h].index[
                                    self.complete_dataset.loc[:, col_name_h] >= 0]]
                                non_neg_index_min = [*self.complete_dataset.loc[:,
                                                      col_name_h].index[self.complete_dataset.loc[:, col_minutes] >= 0]]

                                # if not non_neg_index_hours == non_neg_index_min:
                                #     print('sdsd')
                                #     pass

                                # add transformed hours to minutes with the minutes and store in the hours column
                                self.complete_dataset.loc[non_neg_index_min,
                                                          col_name_h] = self.complete_dataset.loc[non_neg_index_min,
                                                                                                  col_minutes] + col_hour_to_min
                                # convert the minutes to hours to reduce the cardinality
                                # col_name_minutes_idx = np.where(np.asanyarray(col_hour_to_min > 0.0))[0].tolist()
                                self.complete_dataset.loc[non_neg_index_hours,
                                                          col_name_h] = self.complete_dataset.loc[non_neg_index_hours,
                                                                                                  col_name_h] / 60

                                # # missing code definitions are change to zero
                                # self.complete_dataset.loc[self.complete_dataset[col_name_h] < 0, col_name_h] = 0

                                # log the minutes column that will be drop
                                droped_minutes[col_minutes] = self.dataset_dictionary[
                                    self.dataset_dictionary['Column Name'] == col_minutes
                                    ]['Question Name (Abbreviated)'].values[0]
                                # drop the minutes column
                                self.complete_dataset.drop(labels=col_minutes, axis=1, inplace=True)
                                print(f'\ndropped: {col_minutes}')
                                modified_hours.append(col_name_h)
                            else:
                                non_neg_index_min = [*self.complete_dataset.loc[:, col_name_h].index[
                                    self.complete_dataset.loc[:, col_minutes] >= 0]]
                                # min_nan_idx = np.where(np.asanyarray(np.isnan(
                                #     self.complete_dataset.loc[:,col_minutes])))[0].tolist()
                                # self.complete_dataset.loc[:,col_minutes].fillna(value=0, inplace=True, axis=0)
                                self.complete_dataset.loc[non_neg_index_min, col_minutes] = self.complete_dataset.loc[:,
                                                                                            col_minutes] + col_hour_to_min
                                # self.complete_dataset.loc[:, col_minutes] = self.complete_dataset.loc[:,
                                #                                             col_minutes].astype(int)
                                # log the hours column that will be drop
                                droped_minutes[col_name_h] = self.dataset_dictionary[
                                    self.dataset_dictionary['Column Name'] == col_name_h
                                    ]['Question Name (Abbreviated)'].values[0]
                                self.complete_dataset.drop(labels=col_name_h, axis=1, inplace=True)
                                print(f'\ndropped: {col_name_h}')
                                modified_minutes.append(col_minutes)
                            # assert hours_to_min_nan_idx == min_nan_idx # -> True
                        else:
                            print(f'\nUnable to locate columns respective minute column {col_minutes}')
                    else:
                        print(f'\nUnable to locate columns {col_name_h} in complete dataset')
                else:
                    continue

        if keep_hour_format:
            print(f'\nRemoved minutes. Converted hours to minutes and then to hours. Droped minutes question = ')
            for key, val in droped_minutes.items():
                print(f'\n{key} : {val}')
        else:
            print(f'\nRemoved hours and converted to minutes to later unify to the resp. minute question = ')
            for key, val in droped_hours.items():
                print(f'\n{key} : {val}')

    def _convert_height_meters_and_pounds_to_kg(self):
        """
        dem_0600 and dem_0610 are height in feet and inches. Merge to convert in meters and remove the latter.
        :return:
        """
        dropped_height = {}
        assert 'dem_0600' in [*self.complete_dataset.columns]
        assert 'dem_0610' in [*self.complete_dataset.columns]

        feet_meters = self.complete_dataset['dem_0600'] / 3.281
        inches_meters = self.complete_dataset['dem_0610'] / 39.37

        hight_metrics = feet_meters + inches_meters

        self.complete_dataset.drop(labels=['dem_0600', 'dem_0610'], axis=1, inplace=True)

        dropped_height['dem_0600'] = True
        dropped_height['dem_0610'] = True

        self.complete_dataset['dem_hight_meters'] = hight_metrics
        print(f'\nRemoved feet and inches height ("dem_0600", "dem_0610") and created only '
              f'\nmeters predictor dem_hight_meters')
        # Apply a treshold to the hight, if it surpass 2 meters it will be set to 2 meters
        # self.complete_dataset['dem_hight_meters'][self.complete_dataset['dem_hight_meters'] > 2.00] = 2.00
        self.complete_dataset.loc[:, 'dem_hight_meters'].clip(lower=None, upper=2.00, axis=0, inplace=True)
        print('Applied tresthold in meters: 2 meters max ')

        # pound to kgs
        index = self.complete_dataset['dem_0700'].index[self.complete_dataset['dem_0700'].apply(np.isnan)]

        self.complete_dataset['dem_0700'] = self.complete_dataset['dem_0700'] / 2.205
        self.complete_dataset['dem_0700'] = self.complete_dataset['dem_0700'].astype('int32')
        print('Pounds converted to Kgs column: dem_0700')

    def _remove_next_shift_questions(self):
        """
        Questions with next shist will be removed are they are not deemed as signficat for the AHI study.
        It is important to first run the function self._hours_to_minutes().
        The mayority of this questions have a very low frequency response percentage, they're not meaningful
        :return:
        """
        print(f'\n Shift question dropped: ')
        for question_ in self.dataset_dictionary['Question Name (Abbreviated)'].items():
            col_name = self.dataset_dictionary.loc[question_[0], 'Column Name']
            question_ = question_[1]
            if isinstance(question_, str):
                # if ' (Next Shift)' in question_ or ' (3rd Shift) ' in question_:
                if ' (3rd Shift) ' in question_:
                    if col_name in [*self.complete_dataset.columns]:
                        print(f'\n {col_name} : {question_}')
                        self.complete_dataset.drop(labels=col_name, axis=1, inplace=True)

    def _additional_categorical_ordinal_dimension(self):
        """
        Many categorical questions have 0 presenting the options:
            No, Never, Not at all, No RLS symptoms, Rarely/Never , No previous Dx
        for this options it should be feasable to replace the nans values by zero
        :return:
        """
        MODIFICABLE_NANS_TO_ZEROS = ['0=No,', '0=Never,', '0=Not at all', '0=No RLS symptoms', '0=Rarely/Never',
                                     '0=No previous Dx,']
        self.modified_answ_path = \
            self.root.joinpath(pathlib.Path(r'data/pre_processed/manual_column_removal/log_modified_answer_response'))
        modified_columns = {}
        print(f'\nNANs replaces by zeros')
        for question_ in self.dataset_dictionary['Numeric Scoring Code '].items():
            col_name = self.dataset_dictionary.loc[question_[0], 'Column Name']
            question_ = question_[1]
            # question_= '0=No previous Dx, 1=Sleep apnea, 2=Insomnia, 3=Circadian Rhythm Disturbance, 4=RLS/PLM, 5=Narcolepsy, 6=RBD, 7=Sleepwalking, 8=Other, 9=Hypersomnia, 10=Bruxism'
            # col_name = 'mdhx_0120'
            if isinstance(question_, str):
                for mod_zeros_ in MODIFICABLE_NANS_TO_ZEROS:
                    if mod_zeros_ in question_:
                        if col_name in [*self.complete_dataset.columns]:
                            non_nan_count = self.complete_dataset[col_name].count() / \
                                            self.complete_dataset[col_name].shape[0]
                            print(f'\n{col_name} : {non_nan_count} : {question_}')
                            modified_columns[col_name] = {
                                'freq_response': non_nan_count,
                                'question': question_,
                                'action': 'nan == 0 . Nans replaces by zerO'
                            }
                            self.complete_dataset.loc[:, col_name].fillna(value=0, inplace=True)
                        else:
                            continue
                            # print(f'\n{col_name} not in dataset')
        modified_columns = pd.DataFrame(modified_columns).T
        modified_columns.to_excel(excel_writer=self.modified_answ_path / 'nan_to_zero_questions.xlsx')
        print(f'\nColumns with 0 as negatives responses, nan values mapped to zeros {modified_columns}')

    def create_new_dictionary_table(self):
        """
        Useful for the Latex document. Make a final table with the renamed columns, with their description and response
        frequency precentage. This way we can know how much bias we induce per question and if the variance was
        highly increased for questions with low response frequency.
        :return:
        """

        table_structure = []
        for dict_col_name_ in self.dataset_dictionary['Column Name'].items():
            col_name = dict_col_name_[1]
            num_scoring = self.dataset_dictionary.loc[dict_col_name_[0], 'Numeric Scoring Code ']
            if col_name in [*self.complete_dataset.columns]:
                # dataset column is in the dictionary definition
                non_nan_count = self.complete_dataset[col_name].count() / self.complete_dataset[
                    col_name].shape[0]
                table_structure.append(
                    {'column_name': col_name,
                     'numeric scoring': num_scoring,
                     'Fequency Response (Percentage)': non_nan_count
                     })
        table_structure = pd.DataFrame.from_records(table_structure)

        # make latex format file

    def _missing_data_code_mapper(self, replace_nan_negative=Optional[bool] == False):
        """
        Function to work only with negative responses:
            -99 = question not assigned
            -88 = participant responses never
            -66 =  not selected
            -55 = do not know
            -44 = not applicable
            -33 = insufficient data
        I fwe leave this negative values the variance of the quesions will increase since the questions options are in a
        low range of values e.g [0, 8]. Consequently, we will re-map this values to negative number.
        Somethinf to consider is that the negative responses can also be in a string formar, they are not all integer.
        We must conver the dtype while considering the presence of nan values. Float format then.

        Why don't we apply the same lofic for the missing values? The nans will be replaced by negatives, which will
        make teh response be outside the definition.

        Missing values can be treated as a separate category by itself. We can create another category for the
        missing values and use them as a different level. This is the simplest method.
        https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
        bu.edu/sph/files/2014/05/Marina-tech-report.pdf

        :param replace_nan_negative:
        :return:
        """
        mapper = {-99.0: -7,
                  -88.0: -2,
                  -66.0: -6,
                  -55.0: -5,
                  -44.0: -4,
                  -33.0: -3}
        nan_mapper = -1

        print(f'\nMapper applied to values in frame to reduce cardinality of missing_code (old_val, new) = \n{mapper}')
        for col_ in [*self.complete_dataset.columns]:
            self.complete_dataset[col_].replace(to_replace=mapper, inplace=True)
        if replace_nan_negative:
            print(f'\nIncluded nan category as {nan_mapper}')
            self.complete_dataset.fillna(nan_mapper, inplace=True)

    def _decode_listcol_and_screen_subjects(self, apply_one_hot: [Optional] == True,
                                            report: Optional[bool] == True,
                                            remove_screening_columns:Optional[dict] = None):
        """
        Convert multiple answers of  format ["1","3","4","5"] to list of integers.
        remove rows based on values in the multiple response questions.
        apply one hot to the questions

        Multiple answers that have screening questions which we want to remove will be passed as a dictionary where the
        kwy is the column name and the value the option we want to remove
        :param apply_one_hot:
        :param report:
        :param remove_screening_columns:
        :return:
        """
        nonpredictors = ['subject', 'ahi']
        predictors = [x for x in [*self.complete_dataset.columns] if x not in nonpredictors]
        removed_screening_subjects = {}
        subject_to_remove_idx = []
        for col_ in tqdm(predictors, desc="Removing Screening subjects", colour="green"):
            # self.complete_dataset[col_].str.contains('"').any()
            self.complete_dataset[col_] = self._decode_strlist_to_int(sample=self.complete_dataset[col_], name=col_)
            # remove rows based on values in the multiple response questions
            if not (remove_screening_columns is None):
                # remove columns dictionary is present
                if col_ in [*remove_screening_columns.keys()]:
                    # if col_ is in the questions to act, iterate over the column of lists
                    for sam_ in self.complete_dataset[col_].iteritems():
                        # check if there is an intersection between the answered responses and the screening questions
                        screening_answer_in_row = list(set(sam_[1]) & set(remove_screening_columns[col_]))
                        if len(screening_answer_in_row) > 0:
                            # a patient answered a screening questions for removal
                            for screen_in_row_ in screening_answer_in_row:
                                # Iterate over all the screening answers we want to remove in col_
                                subject_to_remove_idx.append(sam_[0])
                                # log the removal to later save in a file as a frame
                                removed_screening_subjects[self.complete_dataset[nonpredictors[0]][sam_[0]]] = {}
                                removed_screening_subjects[self.complete_dataset[nonpredictors[0]][sam_[0]]] = {
                                    'Screening_question': col_,
                                    'Screening_options':screen_in_row_
                                }
        if not (remove_screening_columns is None):
            # aApply the drop, reset index, and save the logs of the removed subjects
            self.complete_dataset.drop(labels=subject_to_remove_idx, axis=0, inplace=True)
            self.complete_dataset.reset_index(drop=True, inplace=True)
            if report:
                print(
                    f'\nScreening subject removed: {self.complete_dataset["subject"][subject_to_remove_idx]}')
            # logs of the removed subjects
            removed_screening_subjects = pd.DataFrame(removed_screening_subjects).T
            removed_screening_subjects.to_excel(
                excel_writer=self.modified_answ_path / 'removed_screening_subjects.xlsx')
            if report:
                print(f'\nScreening subject removed saved in '
                      f'{self.modified_answ_path /"removed_screening_subjects.xlsx"}')
        # %% Apply one hot encoding after the screening filter has been implemented
        if apply_one_hot:
            for col_ in tqdm(predictors, desc="Applying one hot encoding to the list columns", colour="blue"):
                if any([1 for lst in self.complete_dataset[col_] if isinstance(lst, list)]):
                    self.list_type_questions.append(col_)
                    # we have the list type column and we have selected to apply the one hot encoding
                    # Create the dummy df with empty values, columns are all the possible values for col
                    df_empty = pd.DataFrame(0, columns=self.asq_dictionary[col_].keys(),
                                            index=self.complete_dataset[nonpredictors[0]])
                    # populate the dummy df with a one-hot structure
                    for idx_row, row in enumerate(self.complete_dataset[nonpredictors[0]]):
                        for num in [*self.asq_dictionary[col_].keys()]:  # Iterate over the columns
                            if self.complete_dataset[col_][idx_row] == self.complete_dataset[col_][idx_row]:
                                # not a nan. nans have a property -> nan != nan
                                if num in self.complete_dataset[col_][idx_row]:  #the num is a respon's answer
                                    df_empty.loc[row, num] = 1
                                # else:
                                #     df_empty.loc[row, num] = 0
                    # create the dictionary mapper for rename
                    columns_mapper = {}
                    for dumcol in [*df_empty.columns]:
                        columns_mapper[dumcol] = col_ + '_' + str(self.asq_dictionary[col_][dumcol]) + '_' + str(dumcol)

                    # Rename the columns with the use of the dictionary
                    df_empty.rename(columns=columns_mapper, inplace=True)
                    # df_empty.reset_index(drop=True, inplace=True)  -> check this method
                    df_empty.reset_index(inplace=True)
                    # merge the dataframe with the complete one
                    self.complete_dataset = pd.merge(right=df_empty, left=self.complete_dataset, how='inner',
                                                     on=nonpredictors[0])
                    del df_empty
                    # drop the original column
                    self.complete_dataset.drop(labels=col_, inplace=True, axis=1)
                    if report:
                        print(f'\nOne hot applied to {col_} with mapping: \n{columns_mapper}')

    def _review_screeings(self):
        """
        From the screenong questions, some good analysis can be made for example patients which previusly where
        diagnosed with Apnea (mdhx_0120 =1), present in the study an AHI<15. Furthermore, some of this patients state
        that they are not under a current treatment mdhx_0300=0
        :return:
        """
        # get row index where mdhx_012  0 = 1
        col_ = 'mdhx_0120'
        row_idx_mdhx_0120_1 =[]
        for row_ in self.complete_dataset[col_].iteritems():
           if 1 in  row_[1]:
               row_idx_mdhx_0120_1.append(row_[0])

        # frame where mdhx_0120 =1 is present:
        print(self.complete_dataset.loc[row_idx_mdhx_0120_1,['mdhx_0300', 'ahi']])
        frame = self.complete_dataset.loc[row_idx_mdhx_0120_1,['mdhx_0300', 'ahi']]
        frame_lower_ahi = frame[frame['ahi'] < 15]
        # How many in the frame_lower_ahi are using meds (mdhx_0300=1)?
        col_ = 'mdhx_0300'
        row_idx_mdhx_0300_1= []
        for row_ in frame_lower_ahi[col_].iteritems():
           if 1 in  row_[1]:
               row_idx_mdhx_0300_1.append(row_[0])

        frame_lower_ahi.loc[row_idx_mdhx_0300_1,:]

        frame_lower_ahi.iloc[24, :]


    @staticmethod
    def _decode_strlist_to_int(sample: pd.Series, name:str):
        """ convert columns wirth rows ["1","3","4","5"] to [1, 3, 4, 5] """

        # sample = self.merged[col][3]
        def _str_lst_func(x, **kwargs):
            if pd.isna(x):
                x = np.nan
                return x
            if isinstance(x, float) or isinstance(x, int) or isinstance(x, list):
                return x
            elif isinstance(x, str):
                if ':' in x:
                    # time format, ignore
                    return x
                else:
                    x = x.replace('"', '').replace('[', '').replace(']', '').split(',')
                    x = [int(sam) for sam in x]
                    # print(kwargs)
                return x

        prior_shape = sample.shape[0]
        sample = sample.apply(func=_str_lst_func, name=name)

        if sample.shape[0] == prior_shape:
            return sample
        else:
            raise ValueError(f'\nIn method decode_strlist_to_int. Length do not match, not all values were converted')

    def _dtypes_float_to_int(self):
        """
        Once there are no more nan values we can convert the frame columns that have only ints in float format to
        int format
        :return:
        """
        for col_ in tqdm([*self.complete_dataset.columns]):
            if not any([1 for val in self.complete_dataset[col_] if isinstance(val, str)]):
                # num_floats = [j for val in self.complete_dataset[col_] if isinstance(val, float)
                #               for j in range(val) if j.is_integer()]
                num_floats = []
                for val in self.complete_dataset[col_]:
                    if isinstance(val, float):
                        if val.is_integer():
                            num_floats.append(val)
                if len(num_floats) == self.complete_dataset.shape[0]:
                    # if all are integer then the len == df.shape, else there are float
                    self.complete_dataset[col_] = self.complete_dataset[col_].astype('int32')

    @staticmethod
    def _decode_time_to_decimal(sample) -> pd.Series:
        """
        decimal time
        Transform the string with structure '20:00:00' to floats. Seconds resolution is not necessary, we only need
        hours and minutes. So we can make a scale of range [0:24] where the decimals are the minutes
        :param sample: Series from self.merged, column where we have data format values
        :return:
        """
        for idx, row in enumerate(sample):
            if not pd.isnull(row):
                hour, min = row.split(':')[0:2]
                sample[idx] = [int(hour) + (float(min) / 100)][0]
        return sample

    def _decode_time_cyclical(self, all_minutes: Optional[bool] == True):
        """Using the cyclical transformation. We would need to add more columns to the dataframe sched_0700."""
        hours_in_day = 24
        minutes_in_day = hours_in_day * 60
        PLOT = False
        if all_minutes:
            # comvert the hours to minutes, sum with the minutes and transform as a single predictor
            sin_min, cos_min, time_index_frame = [], [], []
            for col_ in [*self.complete_dataset.columns]:
                # col_ = 'sched_0700'
                if col_ in [*self.asq_dictionary.keys()]:
                    if self.asq_dictionary[col_] == 'time':
                        for idx_, sam_ in self.complete_dataset[col_].items():
                            if isinstance(sam_, str):
                                if sam_ == sam_:  # if it is not a nan
                                    if ':' in sam_:
                                        sam = sam_.split(':')
                                        sam = [int(time_) for time_ in sam]  # hour minutes seconds
                                        assert len(sam) == 3
                                        sam[0] = sam[0] * 60  # hours to minutes
                                        sam[1] = sam[0] / 60  # seconds to minutes
                                        time_minutes = int(sum(sam))

                                        sin_min.append(np.sin(2 * np.pi * time_minutes / minutes_in_day))
                                        cos_min.append(np.cos(2 * np.pi * time_minutes / minutes_in_day))

                                        time_index_frame.append(idx_)

                        if PLOT:
                            figure_size = (60, 30)
                            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figure_size, dpi=300)
                            plt.plot(cos_min, color='magenta', marker='o', mfc='pink')
                            plt.show()

                        # asing the time format while conserving the missing values
                        cos_min_series = self.complete_dataset[col_].copy()
                        cos_min_series[time_index_frame] = cos_min
                        cos_min_series.name = col_ + '_' + 'cos'

                        sin_min_series = self.complete_dataset[col_].copy()
                        sin_min_series[time_index_frame] = sin_min
                        sin_min_series.name = col_ + '_' + 'sin'

                        print(f'\nInserting new cos and sine time columns = '
                              f'\n{cos_min_series.name, sin_min_series.name}')
                        self.complete_dataset = pd.concat([self.complete_dataset, cos_min_series,
                                                           sin_min_series], axis=1)

                        print(f'\nRemoving all time column = {col_}')
                        self.complete_dataset.drop(labels=col_, axis=1, inplace=True)

    # %% methods for the class

    def open_complete_dataset_dictionary(self):
        """
        Open the dictionary that contains only the questions available in the complete dataset. The dictionary is a csv
        file with a frame structure and contains the columns:['Table Name', 'Column Name', 'Numeric Scoring Code ',
        'Question Name (Abbreviated)', 'remove', 'action'].
        :return:
        """
        self.dataset_dictionary = pd.read_excel(io=self.table_dict_manual_removal_path)
        self.dataset_dictionary.drop(labels='Unnamed: 6', axis=1, inplace=True)
        del self.table_dict_manual_removal_path

    def manual_removal_from_table(self):
        """
        From the open_complete_dataset_dictionary we will use the manualy annotated ones in the remove column to remove
        columns from the complete_dataset.
        :return:
        """
        print(f'\nManually marked for removal:')
        for question_ in self.dataset_dictionary['remove'].items():
            col_name = self.dataset_dictionary.loc[question_[0], 'Column Name']
            question_ = question_[1]
            if int(question_) == 1:
                # column marked for removal
                if col_name in [*self.complete_dataset.columns]:
                    non_nan_count = self.complete_dataset[col_name].count() / self.complete_dataset.shape[0]
                    print(f'\n{col_name}; freq_response: {non_nan_count}')
                    self.complete_dataset.drop(labels=col_name, axis=1, inplace=True)

    def filter_by_threshold(self, remove_bool: Optional[bool] == False):
        """
        To keep a unbaise dataset and not reducing the variance of the dataset it is important to see if it is worth
        it to replace the nan values of the questions or to remove them.
        :return:
        """
        # define a response frequency
        answ_freq_threhsold = 0.7

        # store the removed questions in a dictionary
        removed_questions = {}
        # list of all question tables from the created dictionary
        output_file = r'C:\Documents\msc_thesis_project\data\pre_processed' \
                      r'\manual_column_removal\table_dict_intersection_complete_dataset_with_stages.xlsx'
        dataset_dictionary = pd.read_excel(io=output_file)
        dataset_dictionary.drop(labels='Unnamed: 6', axis=1, inplace=True)

        table_name = [tab.split('_')[0] for tab in dataset_dictionary['Column Name']]
        table_name = list(sorted(set(table_name)))

        def answer_frequency(dataset: pd.DataFrame, table_name: str, removed_questions: dict,
                             remove: Optional[bool] == False):
            """
            To keep a unbaise dataset and not reducing the variance of the dataset it is important to see if it is worth
            it to replace the nan values of all questions. We can define a threshold for the answer frequency of the
            questions. If the answer frequency of the question is not surpass we can delete it of the dataset.
            To delete we must set  remove=True
            :param dataset:
            :param table_name:
            :param removed_questions:
            :param remove:
            :return:
            """
            question_from_tab = [col_ for col_ in [*dataset.columns] if table_name in col_]
            removed_questions['sched'] = []
            # answer frequency
            print(f'\n___Answer frequency: {table_name} ______________________')
            for col_ in question_from_tab:
                non_nan_count = dataset[col_].count() / dataset.shape[0]
                # print(f'\n  {col_} = {non_nan_count}')
                if non_nan_count < answ_freq_threhsold:
                    # remove questions if the answer frequency is lower than the treshold
                    if remove:
                        print(f'\n Column {col_} removed with answer frequency {non_nan_count}')
                        removed_questions['sched'].append(col_)
                        dataset.drop(labels=col_, inplace=True, axis=1)
                        print(f'\nTotal removed from sched class {len(removed_questions["sched"])}')
                    else:
                        print(f'\n Column candidate to remove {col_}  with answer frequency {non_nan_count}')
            return removed_questions

        for col_ in table_name:
            removed_questions[col_] = answer_frequency(dataset=self.complete_dataset, table_name=col_,
                                                       removed_questions=removed_questions, remove=remove_bool)

    def normalize_feature(self):
        # https://stats.stackexchange.com/questions/385775/normalizing-vs-scaling-before-pca
        # https://stats.stackexchange.com/questions/399430/does-categorical-variable-need-normalization-standardization
        # https://stats.stackexchange.com/questions/359015/ridge-lasso-standardization-of-dummy-indicators

        # if we check the min and max
        def minMax(x):
            return pd.Series(index=['min', 'max'], data=[x.min(), x.max()])

        min_max_column = self.complete_dataset.apply(minMax)

        # not much variation is present as we have categorical variabesl
        # the time predictors are sine and cosines with the y axis limited between [-1, 1]

        # normalize age dem_0110
        def normalize_column(values):
            min = np.min(values)
            max = np.max(values)
            norm = (values - min) / (max - min)
            return pd.DataFrame(norm)

        self.complete_dataset["dem_0110"] = normalize_column(self.complete_dataset["dem_0110"])

    # %% Getters
    def get_results(self):
        if hasattr(self, 'complete_dataset'):
            print(f'\nReturning frame of dimension {self.complete_dataset.shape}')
            return self.complete_dataset
        else:
            return pd.read_csv(filepath_or_buffer=self.output_path, index_col=0)

    def open_complete_dataset(self):
        """Extract the asq answered questionnaire from the class asq_added_entries"""
        complete_dataset_path = self.root.joinpath(pathlib.Path(self.config['pre_processed_merged']) /
                                                   self.config['complete_dataset_name'])
        if complete_dataset_path.exists():
            self.complete_dataset = pd.read_csv(filepath_or_buffer=complete_dataset_path, low_memory=False)
            print(f'\n Extracted merged ASQ from {complete_dataset_path}')
            print(f'\n Dimenions: {self.complete_dataset.shape}')
            return self.complete_dataset
        else:
            ValueError(f'\n Unable to open complete dataset')

    def save_frame(self):
        """
        Save the dataframe in the pre-processing directory
        :return:
        """
        if hasattr(self, 'complete_dataset'):
            print(f'\nSaving frame to directory {self.output_path}')
            self.complete_dataset.to_csv(path_or_buf=self.output_path)
        else:
            raise ValueError(f'\nAttribute "complete_dataset" not found for saving')

    def check_complete_dataset(self, override: Optional[bool] == False) -> bool:
        """
        Check if the complete_dataset created on this class is already existing
        :return:
        """
        if override:
            # not run the class and override the already existing frame
            return False
        else:
            if self.output_path.exists():
                print(f'\nComplete dataset existing in {self.output_path}')
                return True
            else:
                print(f'\nComplete dataset not found in {self.output_path}')
                return False


    def apply_days_threshold(self, days_threshold_diff:int=360):
        """
        Apply a threshold difference to the asq datset
        :param days_threshold_diff:
        :return:
        """
        if not hasattr(self, 'complete_dataset'):
            raise ValueError(f'\n self.complete_dataset is not present in the class')

        # read the frame with the date difference
        self.asq_date_frame_path = self.root.joinpath(self.config['match_subject_ahi_path'],
                                                      'subject_asq_ahi_delta.xlsx')
        asq_date_frame = pd.read_excel(self.asq_date_frame_path)
        asq_date_frame.set_index(keys=['Unnamed: 0'], drop=True, inplace=True)

        initial_shape = asq_date_frame.shape[0]
        print(f'Initial shape: {initial_shape}')

        if days_threshold_diff:
            subjects_to_drop_frame = asq_date_frame[asq_date_frame['delta'] > days_threshold_diff]
            print(f'\n Days threshold applied, subjects removed: {subjects_to_drop_frame.shape[0]}')
            subs_drop = []
            for subj_ in [*subjects_to_drop_frame['subject']]:
                if subj_ in [*self.complete_dataset['subject']]:
                    # get the index of rows in a  DataFrame whose column matches specific values:
                    subs_drop.append(self.complete_dataset.index[self.complete_dataset['subject'] == subj_].tolist()[0])

            if not [*self.complete_dataset.loc[subs_drop, 'subject']].sort() ==\
                   [*subjects_to_drop_frame['subject']].sort():
                raise ValueError(f'\nERROR intersections subjects are different, removing wrong subjects!')
            self.complete_dataset.drop(labels=subs_drop, inplace=True, axis=0)
            self.complete_dataset.reset_index(drop=True, inplace=True)
            print(f'\n Days threshold of {days_threshold_diff} have been applied')



# %% main
if __name__ == "__main__":
    dataset_info = ASQ_Reader_Complete_Dataset(config=config, override_existing=True, days_threshold=365)
    complete_dataset = dataset_info.get_results()
    print(complete_dataset)
