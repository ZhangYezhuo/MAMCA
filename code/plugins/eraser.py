'''
Experiment Result      Archiver         and SavER (ERASER)

EEEEEEEE  RRRRRR          A        SSSSSSSS  EEEEEEEE  RRRRRR
E         R     R        A A       S         E         R     R
EEEEEEEE  RRRRRR        AAAAA      SSSSSSSS  EEEEEEEE  RRRRRR
E         R    R       A     A            S  E         R    R
EEEEEEEE  R     RR    A       A    SSSSSSSS  EEEEEEEE  R     RR
'''

import pandas as pd
import os
import pickle
import datetime
import re

'''
Each crumble manages a CSV file. Specifically, a crumble manages columns through dictionaries, and each column manages rows through dictionaries.
'''
def get_time():
        year, month, day, hour, min, sec = re.split("-| |:", str(datetime.datetime.now()))
        return "{}{}{}_{}{}{}".format(year, month, day, hour, min, int(float(sec)))

class Crumble(object):
    def __init__(self, root = None, name = "crumble") -> None:
        self.root = root
        self.name = name
        self.col = dict()
        self.keys = list()# keys针对column
        self.row_keys = list()

        self.csv = pd.DataFrame().from_dict(dict())
        self.cvs_path = "{0}{1}.csv".format(self.root, self.name)
        self.csv.to_csv(self.cvs_path, index=True)

    '''
    Columns and rows manage the data in rows and columns, respectively. A column manages rows as a dictionary, and a row manages data as a dictionary indexed by time.
    '''
    class Row(object):
        def __init__(self, value) -> None:
            self.dict = dict()
            self.dict[get_time()]=value
        
        def __str__(self):
            return str(self.dict)

    class Column(object):
        def __init__(self) -> None:
            self.keys = list()
            self.row = dict()

        def __getitem__(self, key):
            if not key in self.keys:
                self.keys.append(key)
            return self.row[key]
        
        def __setitem__(self, key, value):
            if not key in self.row.keys():
                if not key in self.keys: self.keys.append(key)
                self.row[key] = Crumble.Row(value)
            else:
                self.row[key].dict[get_time()]=value
    
    '''
    create a new crumble from a CSV file
    '''
    def read(self, path):
        self.clear(confirm=True)
        # self.csv_path = path
        self.csv = pd.read_csv(path, index_col=0).copy()
        self.keys = self.csv.columns.values.tolist()
        self.row_keys = self.csv.index.values.tolist()
        for key in self.keys:
            for row_key in self.row_keys:
                self[key][row_key] = self.csv[key][row_key].copy()
                #     self.col[key].row[row_key] = Crumble.Row(self.csv[key][row_key])

    '''
    Crumble writes the data prepared in columns and rows to the specified CSV file.
    '''
    def write(self):
        col_indexs = self.keys
        temp = list()
        for key in self.keys:
            for subk in self.col[key].keys: temp.append(subk)
        self.row_keys = list(set(temp))
        row_indexs = self.row_keys
        self.csv = pd.DataFrame(index = row_indexs, columns = col_indexs)
        for key in self.keys:
            for subk in self.row_keys:
                try: temp = self.col[key][subk].dict[list(self.col[key][subk].dict.keys())[-1]]
                except: temp = None
                self.csv.loc[subk, key] = temp 
        self.csv.to_csv(self.cvs_path, index=True)

    '''
    crumble clear
    '''
    def clear(self, confirm = False):
        if confirm:
            self.col = dict()
            self.keys = list()
            self.row_keys = list()
            self.csv = pd.DataFrame().from_dict(dict())
            self.csv.to_csv("{0}{1}.csv".format(self.root, self.name))

    def __getitem__(self, key):
        if not key in self.col.keys():
            self.col[key] = Crumble.Column()
        if not key in self.keys: self.keys.append(key)
        
        return self.col[key]

    def __setitem__(self, key, value):
        self.col[key] = value

'''
Eraser oversees the management of crumbles
'''
class Eraser(object):
    def __init__(self, root = None, name = None) -> None:
        self.root = root
        self.name = name
        self.crumble = dict()
    
    '''
    save eraser to local pkl file
    '''
    def save(self, name=None):
        if name != None: self.crumble[name].write()
        if not os.path.exists(self.root): os.makedirs(self.root)
        with open("{0}{1}.pkl".format(self.root, self.name),"wb") as file:
            pickle.dump(self, file, True)

    '''
    load from local pkl 
    '''
    def load(self, path=None):
        with open(path, "rb") as file:
            temp = pickle.load(file)
            self.root = temp.root
            self.name = temp.name
            self.crumble = temp.crumble
    
    '''
    add a crumble for eraser
    '''
    def add_crumble(self, name):
        if not name in self.crumble.keys():
            print("create a new crumble")
            self.crumble[name] = Crumble(self.root, name)
            self.crumble[name].clear(confirm = True)
        else: print("the crumble has existed")
        return self.crumble[name]