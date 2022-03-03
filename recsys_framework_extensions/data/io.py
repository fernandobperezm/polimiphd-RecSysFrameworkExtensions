#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27/04/2019

@author: Maurizio Ferrari Dacrema
"""
from __future__ import annotations

import json
import os
import shutil
import warnings
import zipfile
from enum import Enum
from typing import Any, Type

import numpy as np
import pandas as pd
import scipy.sparse as sps

from Recommenders.DataIO import DataIO as RecSysFrameworkDataIO


def attach_to_extended_json_decoder(enum_class: Type[Enum]):
    """
    Examples
    --------
    >>> from recsys_framework_extensions.data.io import attach_to_extended_json_decoder
    >>> from enum import Enum

    >>> @attach_to_extended_json_decoder
    >>> class MyEnum(Enum):
    >>>   VALUE = "VALUE"
    """
    ExtendedJSONEncoderDecoder.attach_enum(
        enum_class=enum_class
    )
    return enum_class


class ExtendedJSONEncoderDecoder(json.JSONEncoder):
    """
    Some values cannot be easily serialized/deserialized in JSON. For instance, enums and some
    numpy types.

    ExtendedJSONEncoderDecoder serves as a class to serialize/deserialize those values. The values exported
    by the class might not be portable across architectures, OS, or even programming languages.
    Nonetheless, it provides a very helpful way to serialize objects that otherwise might not be
    saved.
    """
    _ATTACHED_ENUMS: dict[str, Type[Enum]] = dict()

    @staticmethod
    def attach_enum(enum_class: Type[Enum]):
        """
        Examples
        --------
        >>> from recsys_framework_extensions.data.io import ExtendedJSONEncoderDecoder

        >>> from enum import Enum
        >>> class MyEnum(Enum):
        >>>   VALUE = "VALUE"
        >>> ExtendedJSONEncoderDecoder.attach_enum(enum_class=MyEnum)
        """
        enum_name = enum_class.__name__
        if enum_name in ExtendedJSONEncoderDecoder._ATTACHED_ENUMS:
            raise KeyError(
                f"Enum '{enum_name}' has already been attached. This may indicate that you attached this enum before "
                f"or another Enum has this name."
            )

        ExtendedJSONEncoderDecoder._ATTACHED_ENUMS[enum_name] = enum_class

    def default(self, obj: Any) -> Any:
        """
        This is the method that is called when a Python object is being serialized, i.e.,
        when json.dump(..., cls=ExtendedJSONEncoderDecoder) or json.dumps(..., cls=ExtendedJSONEncoderDecoder)
        are called.
        """
        if isinstance(obj, (np.integer, np.int32)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Enum):
            return {"__enum__": str(obj)}

        # We leave the default JSONEncoder to raise an exception if it cannot serialize something.
        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def decode_hook(obj: Any) -> Any:
        """
        This is the method that is called when a str object is being deserialized, i.e.,
        when json.load(..., object_hook=ExtendedJSONEncoderDecoder.decode_hook) or
        json.loads(..., object_hook=ExtendedJSONEncoderDecoder.decode_hook) are called.

        Notice that we have to load in the class all the enums classes, if we do not do this,
        Python will not know how to instantiate every specific enum class (basically we need to
        import them and have them in scope so it can recognize the enum class).
        """
        if "__enum__" in obj:
            name, member = obj["__enum__"].split(".")

            if name not in ExtendedJSONEncoderDecoder._ATTACHED_ENUMS:
                raise KeyError(
                    f"This enum '{name}' has not been attached to the '{ExtendedJSONEncoderDecoder.__name__}' class. "
                    f"Check the decorator '{attach_to_extended_json_decoder.__name__}' or the static method "
                    f"'{ExtendedJSONEncoderDecoder.__name__}.{ExtendedJSONEncoderDecoder.attach_enum.__name__}' documentation to "
                    f"learn how to attach enumerators to this class."
                )

            # This does not instantiate the enum, instead, it gets the attribute ``member`` from the class
            # ``enum_class``, i.e. it is doing ``enum_class.member``
            enum_class = ExtendedJSONEncoderDecoder._ATTACHED_ENUMS[name]
            return getattr(enum_class, member)

        else:
            return obj


class DataIO(RecSysFrameworkDataIO):
    """ DataIO"""

    def __init__(self, folder_path):
        super().__init__(folder_path=folder_path)

    def save_data(self, file_name, data_dict_to_save):
        # If directory does not exist, create with .temp_model_folder
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        if file_name[-4:] != ".zip":
            file_name += ".zip"

        current_temp_folder = self._get_temp_folder(file_name)

        try:

            data_format = {}
            attribute_to_save_as_json = {}

            for attrib_name, attrib_data in data_dict_to_save.items():
                current_file_path = current_temp_folder + attrib_name

                if isinstance(attrib_data, pd.DataFrame):
                    # attrib_data.to_hdf(current_file_path + ".h5", key="DataFrame", mode='w', append = False, format="table")
                    # Save human readable version as a precaution. Append "." so that it is classified as auxiliary file and not loaded
                    attrib_data.to_csv(current_temp_folder + "." + attrib_name + ".csv", index=True)

                    attrib_data.to_parquet(
                        path=current_file_path + ".parquet",
                        engine="pyarrow",
                        compression=None,
                    )

                    # Using "fixed" as a format causes a PerformanceWarning because it saves types that are not native of C
                    # This is acceptable because it provides the flexibility of using python objects as types (strings, None, etc..)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        attrib_data.to_hdf(
                            current_file_path + ".h5",
                            key="DataFrame",
                            mode='w',
                            append=False,
                            format="fixed"
                        )

                elif isinstance(attrib_data, sps.spmatrix):
                    sps.save_npz(current_file_path, attrib_data)

                elif isinstance(attrib_data, np.ndarray):
                    # allow_pickle is FALSE to prevent using pickle and ensure portability
                    np.save(current_file_path, attrib_data, allow_pickle=False)

                else:
                    # Try to parse it as json, if it fails and the data is a dictionary, use another zip file
                    try:
                        _ = json.dumps(attrib_data, cls=ExtendedJSONEncoderDecoder)
                        attribute_to_save_as_json[attrib_name] = attrib_data
                    except TypeError:
                        if isinstance(attrib_data, dict):
                            data_io = DataIO(folder_path=current_temp_folder)
                            data_io.save_data(file_name=attrib_name, data_dict_to_save=attrib_data)
                        else:
                            raise TypeError("Type not recognized for attribute: {}".format(attrib_name))

            # Save list objects
            if len(data_format) > 0:
                attribute_to_save_as_json[".data_format"] = data_format.copy()

            for attrib_name, attrib_data in attribute_to_save_as_json.items():
                current_file_path = current_temp_folder + attrib_name

                # if self._is_windows and len(current_file_path + ".json") >= self._MAX_PATH_LENGTH_WINDOWS:
                #     current_file_path = "\\\\?\\" + current_file_path

                absolute_path = current_file_path + ".json" if current_file_path.startswith(
                    os.getcwd()) else os.getcwd() + current_file_path + ".json"

                assert not self._is_windows or (
                    self._is_windows and len(absolute_path) <= self._MAX_PATH_LENGTH_WINDOWS), \
                    "DataIO: Path of file exceeds {} characters, which is the maximum allowed under standard paths for Windows.".format(
                        self._MAX_PATH_LENGTH_WINDOWS)

                with open(current_file_path + ".json", 'w') as outfile:
                    if isinstance(attrib_data, dict):
                        attrib_data = self._check_dict_key_type(attrib_data)

                    json.dump(attrib_data, outfile, csl=ExtendedJSONEncoderDecoder)

            with zipfile.ZipFile(
                self.folder_path + file_name + ".temp", 'w',
                compression=zipfile.ZIP_DEFLATED
            ) as myzip:
                for file_to_compress in os.listdir(current_temp_folder):
                    myzip.write(current_temp_folder + file_to_compress, arcname=file_to_compress)

            # Replace file only after the new archive has been successfully created
            # Prevents accidental deletion of previous versions of the file if the current write fails
            os.replace(self.folder_path + file_name + ".temp", self.folder_path + file_name)

        except Exception as exec:
            shutil.rmtree(current_temp_folder, ignore_errors=True)
            raise exec

    def load_data(self, file_name):

        if file_name[-4:] != ".zip":
            file_name += ".zip"

        dataFile = zipfile.ZipFile(self.folder_path + file_name)

        dataFile.testzip()

        current_temp_folder = self._get_temp_folder(file_name)

        try:

            try:
                data_format = dataFile.extract(".data_format.json", path=current_temp_folder)
                with open(data_format, "r") as json_file:
                    data_format = json.load(json_file)
            except KeyError:
                data_format = {}

            data_dict_loaded = {}

            for file_name in dataFile.namelist():

                # Discard auxiliary data structures
                if file_name.startswith("."):
                    continue

                decompressed_file_path = dataFile.extract(file_name, path=current_temp_folder)
                file_extension = file_name.split(".")[-1]
                attrib_name = file_name[:-len(file_extension) - 1]

                if file_extension == "csv":
                    # Compatibility with previous version
                    attrib_data = pd.read_csv(decompressed_file_path, index_col=False)

                elif file_extension == "h5":
                    attrib_data = pd.read_hdf(decompressed_file_path, key=None, mode='r')

                elif file_extension == "parquet":
                    attrib_data = pd.read_parquet(
                        path=decompressed_file_path,
                        engine="pyarrow",
                    )

                elif file_extension == "npz":
                    attrib_data = sps.load_npz(decompressed_file_path)

                elif file_extension == "npy":
                    # allow_pickle is FALSE to prevent using pickle and ensure portability
                    attrib_data = np.load(decompressed_file_path, allow_pickle=False)

                elif file_extension == "zip":
                    data_io = DataIO(folder_path=current_temp_folder)
                    attrib_data = data_io.load_data(file_name=file_name)

                elif file_extension == "json":
                    with open(decompressed_file_path, "r") as json_file:
                        attrib_data = json.load(json_file, object_hook=ExtendedJSONEncoderDecoder.decode_hook)

                else:
                    raise Exception(
                        "Attribute type not recognized for: '{}' of class: '{}'".format(
                            decompressed_file_path,file_extension
                        )
                    )

                data_dict_loaded[attrib_name] = attrib_data

        except Exception as exec:
            shutil.rmtree(current_temp_folder, ignore_errors=True)
            raise exec

        shutil.rmtree(current_temp_folder, ignore_errors=True)

        return data_dict_loaded
