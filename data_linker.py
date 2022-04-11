import csv
import random
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyparsing import col
from scipy.signal import correlate

def clean(header: str) -> str:
    return header.replace('"', '').replace("'", '').strip()

def headers_empty(headers: Sequence[str]) -> bool:
    return sum([clean(header) != '' for header in headers]) == 0

def get_headers(file) -> Sequence[str]:
    csv_file = csv.reader(file)
    for line in csv_file:
        if headers_empty(line):
            continue
        return line

    return ()

# What about commas!?
def is_int(s: str) -> bool:
    non_digits = [c for c in s if ord(c) < 48 or ord(c) > 57]
    return not non_digits or (len(non_digits) == 1 and (non_digits[0] == '-' or non_digits[0] == '+'))

# What about commas!?
def is_float(s: str) -> bool:
    non_digits = [c for c in s if ord(c) < 48 or ord(c) > 57]
    if non_digits and non_digits[-1] in ['f', 'd']:
        non_digits = non_digits[:-1]
    if non_digits and ((non_digits[0] == '-' or non_digits[0] == '+') and s.startswith(non_digits[0])):
        non_digits = non_digits[1:]

    return not non_digits or (len(non_digits) == 1 and '.' in non_digits)

def is_bool(s: str) -> bool:
    return s.lower() == 't' or s.lower() == 'f' or s.lower() == 'true' or s.lower() == 'false'

def get_type(s: str) -> bool:
    if is_int(s):
        return np.int64
    if is_float(s):
        return np.float32
    if is_bool(s):
        return np.bool8
    return object

def resolve_type(t1, t2):
    if t1 in [np.int64, np.float32] and t2 in [np.int64, np.float32]:
        return np.float32
    return object

def df_load(path: Path) -> pd.DataFrame:
    with open(path, 'r') as file:
        headers = get_headers(file)

        if headers:
            csv_file = csv.reader(file)

            # Read several lines of headers for all information
            all_values = []
            for line in csv_file:
                if len(line) == len(headers):
                    values = [line[i] for i in range(len(line)) if clean(headers[i]) != '']
                    all_values.append(values)

            #sample_rows = random.sample(all_values, 5)

            #dtypes = [get_type(s) for s in sample_rows[0]]
            dtypes = [get_type(s) for s in all_values[0]]
            for row in all_values:
            #for sample_row in sample_rows[1:]:
                row_dtypes = [get_type(s) for s in row]
                for i, e in enumerate(list(np.equal(dtypes, row_dtypes))):
                    if not e:
                        dtypes[i] = resolve_type(dtypes[i], row_dtypes[i])

            df = pd.DataFrame(all_values, columns=[clean(header) for header in headers if clean(header) != ''])

            for c, dt in zip([header for header in headers if clean(header) != ''], dtypes):
                if dt in [np.int64, np.float32]:
                    df[c] = pd.to_numeric(df[c])


    return df

def get_common_columns(c1: Sequence[str], c2: Sequence[str]) -> Sequence[str]:
    c2_cleaned = [clean(c) for c in c2 if clean(c) != '']

    return [c for c in c1 if clean(c) in c2_cleaned]

def check_value_match(df1: pd.DataFrame, df2: pd.DataFrame, columns: Sequence[str]) -> bool:
    return sum([not df1[c].equals(df2[c]) for c in columns]) == 0

def find_alignment(df1: pd.DataFrame, df2: pd.DataFrame, columns: Sequence[str]) -> bool:
    if len(df2) > len(df1):
        dft = df1
        df1 = df2
        df2 = dft
    
    shifts = []
    # Come back and use correlation/autocorrelation later
    for column in columns:
        offset = np.argmax(np.correlate(df1[column], df2[column], "valid")) + df2[column].iloc[-1] - df1[column].iloc[-1]
        shifts.append((column, offset))

    if sum(np.diff([o for _, o in shifts])) != 0:
        raise ValueError('Found several alignments')

    return shifts[0]

def df_merge(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    common_columns = get_common_columns(df1.columns, df2.columns)

    if len(df1) != len(df2):
        column, offset = find_alignment(df1, df2, common_columns)
        df2 = df2[offset:]
        print(f'Alignment removed {len(df1) - len(df2) + offset} rows')
        df1 = df1[0:len(df2)]
        df1.index = df2.index


    if not check_value_match(df1, df2, common_columns):
        print('These dont look the same. Try again with a threshold?')
        raise ValueError('These dont look the same. Try again with a threshold?')

    remaining_columns = set([c for c in df2.columns if clean(c) != '']) - set(common_columns)

    df = pd.concat([df1, df2[remaining_columns]], axis=1)
    return df
    # Simple merge
    #if len(df1) == len(df2):
        #pass
    # Check for alignment
    #else:
        #pass


p1 = Path('data\\usgs_1900_2027.csv')
p2 = Path('data\\usgs_1900_2027(1).csv')
p3 = Path('inflation_data.csv')

#path = 'usgs_1800_2027.csv'
df1 = df_load(p1)
df2 = df_load(p2)
inflation = df_load(p3)

df3 = df_merge(df1, df2)
df4 = df_merge(df3, inflation)
print(df4)