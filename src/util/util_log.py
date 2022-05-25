"""
Utilities for record logging, path creating and so on.

Created by Zhaoyan @ UCL
"""

from datetime import datetime
import numpy as np
import os
import pandas as pd
import pickle
import json
from pathlib import PurePath, Path
# from treelib import Node, Tree

import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))

class ResultPathManager:
    """

    alias_dict: a dictionary for quick look-up of the paths. Should at least contains following items:
        project_root: Project path is the root of the whole project. Should contain folders such as ./ref, ./src etc.
        working_code: The path to the current executing file.
        results_root: The 'results' folder under the project_root containing all results. The whole folder should under the .ignore file.
        root: The path to the mirror path of working_code path under result_root folder.

    path_tree: A Tree object. The tree is rooted at alias_dict['root']. The node properties are defined as follows:
        tag: (string) the name of current folder
        identifier: (string) the absolute path
        data: (PurePath object) the absolute path

    """
    def __init__(self, project_root_path):
        self.project_root_path = PurePath(project_root_path)
        self.working_code = Path.cwd()
        self.root_path = None

        self.alias_dict = {
            'project_root': self.project_root_path,
            'working_code': self.working_code,
            'results_root': self.project_root_path.joinpath('results'),
        }

        # self.path_tree = Tree()
        self._init_root_path()

    # def _generate_path_tree(self, root_path):
    #     """
    #     Walk in the root_path and update self.path_tree by the path found.
    #     !Note: the paths with/in the folder name start with '_' will be ignored.

    #     :param root_path: string or PurePath object
    #     :return:
    #     """
    #     for root, dirs, files in os.walk(root_path, topdown=True):
    #         parent_path = PurePath(root).parent
    #         folder_name = PurePath(root).parts[-1]
    #         if folder_name[0] == '_':
    #             continue

    #         if self.path_tree.contains(str(parent_path)):
    #             self.path_tree.create_node(tag=folder_name, identifier=str(root), parent=str(parent_path), data=root)
    #         else:
    #             self.path_tree.create_node(tag=folder_name, identifier=str(root), data=root)

    def _init_root_path(self):
        """
        Copy the relative path of the executing file to a new path 'results' under the project root.
        :return:
        """
        relative_path = self.working_code.relative_to(self.project_root_path)
        self.root_path = self.alias_dict['results_root'].joinpath(*relative_path.parts[1:])

        if os.path.isdir(self.root_path):  # If the root path for result already exist.
            print(f'{datetime.now()} I util_log.ResultManager._init_home_path: [Exist] home_path: {self.root_path}')
        else:                              # If the root path for result do not exist.
            os.makedirs(self.root_path, exist_ok=True)
            print(f'{datetime.now()} I util_log.ResultManager._init_home_path: [Create] home_path: {self.root_path}')
        # self._generate_path_tree(self.root_path)
        self.alias_dict['root'] = self.root_path


    def __call__(self, alias):
        """
        Get PurePath object by alias, from self.alias_dict.
        :param alias:
        :return:
        """
        assert alias in self.alias_dict.keys(), f'util_log.ResultPathManager.__call__(): alias not exist: {alias}'
        return self.alias_dict[alias]

    def __iter__(self, alias):  # To test
        self.__call__(alias)



    def add_path(self, new_alias='', parent_alias='', name_new_path=''):
        # Duplication detection
        if new_alias != '':
            assert new_alias not in self.alias_dict.keys(), 'util_log.ResultPathManager.add_path(): duplicated alias name.'
        # Parent exist
        assert parent_alias in self.alias_dict.keys(), f'util_log.ResultPathManager.add_path(): parent_alias not exist: {parent_alias}'

        parent = self.alias_dict[parent_alias]
        new_path = parent.joinpath(name_new_path)
        # self.path_tree.create_node(tag=name_new_path, identifier=str(new_path), data=new_path, parent=str(parent))

        # If the given path had a suffix (such as .json), then do not create a folder.
        if '.' not in str(new_path):
            os.makedirs(new_path, exist_ok=True)


        if new_alias != '':
            self.alias_dict[new_alias] = new_path


    def reset_root(self, new_root_abs_path):
        if isinstance(new_root_abs_path, PurePath):
            self.alias_dict['root'] = new_root_abs_path
        else:
            self.alias_dict['root'] = PurePath(new_root_abs_path)
            os.makedirs(new_root_abs_path, exist_ok=True)

def save_json(path_to_json, dict_to_save):
    assert path_to_json.endswith('.json'), f'{datetime.now()} E util_log.save_json() path_to_json must ends with .json.' \
                                           f'Get {path_to_json} instead.'
    with open(path_to_json, 'w') as f:
        json.dump(dict_to_save, f, indent=4)


def load_json(path_to_json):
    assert path_to_json.endswith('.json'), f'{datetime.now()} E util_log.load_json() path_to_json must ends with .json.' \
                                           f'Get {path_to_json} instead.'
    with open(path_to_json, 'r') as f:
        return json.load(f)


def get_formatted_time():
    """
    Example results: 2020-09-06_22:12:42.307021
    No space. Easier to use in Linux directory commands.
    :return:
    """
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    return time


class Paths:
    """
    The paths to results.
    """
    def __init__(self, result_home_path,
                 record_name,
                 if_log=True,
                 if_weights=True,
                 if_summary=True,
                 if_visualization=True,
                 if_curve=True,
                 if_temp=True,
                 summary_suffix=None):
        """

        :param result_home_path: (String) Path to result home folder. The home folder is usually for a group of setups
                for the same experiment.
        :param record_name: (String) The name of current record, usually is the timestamp.
        :param if_log:
        :param if_weights:
        :param if_summary:
        :param if_visualization:
        :param if_curve:
        :param summary_suffix: (None, String, optional) The suffix to the summary record.
        :return:
        """
        self.result_home_path = result_home_path
        self.record_name = record_name
        self.dict_folder_paths = {
            'folder,log': os.path.join(result_home_path, 'log'),
            'folder,weights': os.path.join(result_home_path, 'weights'),
            'folder,summary': os.path.join(result_home_path, 'summary'),
            'folder,visualization': os.path.join(result_home_path, 'visualization'),
            'folder,curve': os.path.join(result_home_path, 'curve'),
            'folder,temp': os.path.join(result_home_path, 'temp')
        }

        self.dict_file_paths = {
            'file,log': os.path.join(self.dict_folder_paths['folder,log'],
                                     'log.csv'),
            # It is actually a folder
            'file,weights': os.path.join(self.dict_folder_paths['folder,weights'],
                                         f'{record_name}'),
            # It is actually a folder
            'file,summary': os.path.join(self.dict_folder_paths['folder,summary'],
                                         f'{record_name}'),
            # It is actually a folder
            'file,visualization': os.path.join(self.dict_folder_paths['folder,visualization'],
                                               f'{record_name}'),
            'file,curve': os.path.join(self.dict_folder_paths['folder,curve'],
                                       f'{record_name}.npy')
        }

        os.makedirs(result_home_path, exist_ok=True)

        if if_log:
            os.makedirs(self.dict_folder_paths['folder,log'], exist_ok=True)
        if if_weights:
            os.makedirs(self.dict_file_paths['file,weights'], exist_ok=True)
        if if_summary:
            os.makedirs(self.dict_folder_paths['folder,summary'], exist_ok=True)
        if if_visualization:
            os.makedirs(self.dict_file_paths['file,visualization'], exist_ok=True)
        if if_curve:
            os.makedirs(self.dict_folder_paths['folder,curve'], exist_ok=True)
        if if_temp:
            os.makedirs(self.dict_folder_paths['folder,temp'], exist_ok=True)

        self.others = {}

        if summary_suffix:
            self.add_summary_suffix(summary_suffix)

        self.log = self.dict_file_paths['file,log']
        self.weights = self.dict_file_paths['file,weights']
        self.summary = self.dict_file_paths['file,summary']
        self.summary_folder = self.dict_folder_paths['folder,summary']
        self.visualization = self.dict_file_paths['file,visualization']
        self.curve = self.dict_file_paths['file,curve']

        print(f'{datetime.now()} I Path initialization done. '
              f'\tresult_home_path: {result_home_path}'
              f'\trecord_name: {record_name}')

    def add_summary_suffix(self, summary_suffix):
        """

        :param summary_suffix: (String)
        :return:
        """
        temp_old_name = self.dict_file_paths['file,summary'].split('/')[-1]
        self.dict_file_paths['file,summary'] = \
            os.path.join(self.dict_folder_paths['folder,summary'],
                         f'{temp_old_name},{summary_suffix}')
        self.summary = self.dict_file_paths['file,summary']

    def add_other_paths(self, name, path, makefolder=False):
        self.others[name] = path
        if makefolder:
            os.makedirs(path, exist_ok=True)

    def get_all_paths(self):
        all_paths = self.dict_folder_paths
        all_paths.update(self.dict_file_paths)
        all_paths.update(self.others)
        return all_paths


class Log:
    """
    Log the numerical results in a csv file that can be viewed by MS Excel.
    """
    def __init__(self):
        """

        :param timestamp: (String) The timestamp of the record. The timestamp can identify a test record.
        :param path: (String) The path of the .npy file>
        """
        self.log_dict_to_save = {}

    def update(self, key=None, value=None, dict=None):
        """

        :param key: (String)
        :param value: (tuple, list, or scalar for int, float, String values) Notice that the output of the torch models
                        should be wrapped with float() to convert to python scalar.
        :param dict:
        :return:
        """
        if key:
            self.log_dict_to_save[key] = value
        else:
            self.log_dict_to_save.update(dict)

    def save(self, path_log):
        assert path_log.endswith('.csv'),\
            f'{datetime.now()} E path must be pointed at a .csv file. Get {path_log} instead.'
        for key in self.log_dict_to_save.keys():
            if not isinstance(self.log_dict_to_save[key], list):
                self.log_dict_to_save[key] = [self.log_dict_to_save[key]]
            else:
                if len(self.log_dict_to_save[key]) != 1:
                    self.log_dict_to_save[key] = [self.log_dict_to_save[key]]
        if os.path.exists(path_log):
            df = pd.read_csv(path_log)
            df_new = pd.DataFrame(self.log_dict_to_save)
            df = df.append(df_new, sort=False)
        else:
            df = pd.DataFrame(self.log_dict_to_save)
        df.to_csv(path_log, index=False)
        print(f'{datetime.now()} I Log saved:')
        for key, value in self.log_dict_to_save.items():
            print(f'\t{key}: {value}')


class Measurement:
    def __init__(self, name, m_fn=None):
        """

        :param name: (String)
        :param m_fn: (callable) A function or a callable class that can make
        """
        self.name = name

        self.last_value = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.average = 0.0

        self.tape = {}
        self.tape_index = 0
        self.tape_max = None
        self.tape_min = None
        self.m_fn = m_fn

    def clear(self):
        self.last_value = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.average = 0.0

    def update(self, value=None, para_list=None):
        if self.m_fn is not None:
            if para_list is not None:
                value = self.m_fn(value, para_list)
            else:
                value = self.m_fn(value)
        else:
            assert value is not None
        self.last_value = value
        self.sum += value
        self.count += 1
        self.average = self.sum / self.count

    def write_tape(self, plot_index=0):
        """
        One could use the log_results function in util_log.py to save the tapes in csv files.
        :param plot_index: (int, optional) The index for plotting. e.g. epoch index.
        :return:
        """
        self.tape[self.tape_index] = {'name': self.name,
                                      'plot_index': plot_index,
                                      'sum': self.sum,
                                      'count': self.count,
                                      'average': self.average,
                                      'last_value': self.last_value}
        if self.tape_min is None:
            self.tape_min = self.average
        else:
            if self.average < self.tape_min:
                self.tape_min = self.average
        if self.tape_max is None:
            self.tape_max = self.average
        else:
            if self.average > self.tape_max:
                self.tape_max = self.average
        self.tape_index += 1

    def save_tape(self, path_tape, format='json'):
        """

        :param format:
        :param path_tape:
        :return:
        """
        assert format in ['json', 'npy'], f'{datetime.now()} E util_log.Measurement.save_tape(): format only support ' \
                                          f'json, npy; get {format} instead.'
        if format == 'json':
            if path_tape.endswith('.json'):
                with open(path_tape, 'w') as f:
                    json.dump(self.tape, f, indent=4)
            else:
                with open(path_tape + '.json', 'w') as f:
                    json.dump(self.tape, f, indent=4)
        elif format == 'npy':
            if path_tape.endswith('.npy'):
                np.save(path_tape, self.tape)
            else:
                np.save(path_tape + '.npy', self.tape)


def save_dict_to_pickle(dict_obj, path):
    """
    Ref: https://stackoverflow.com/a/19201448
    :param dict_obj:
    :param path:
    :return:
    """
    assert path.endswith('.pkl') or path.endswith('.pickle'), \
        f'{datetime.now()}: util_log.save_dict_to_pickle(): path must ends with .pkl or .pickle. Get {path} instead.'
    with open(path, 'wb') as f:
        pickle.dump(dict_obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict_from_pickle(path):
    """
    Ref: https://stackoverflow.com/a/19201448
    :param path:
    :return:
    """
    assert path.endswith('.pkl') or path.endswith('.pickle'), \
        f'{datetime.now()}: util_log.load_dict_from_pickle(): path must ends with .pkl or .pickle. Get {path} instead.'
    with open(path, 'rb') as f:
        return pickle.load(f)
