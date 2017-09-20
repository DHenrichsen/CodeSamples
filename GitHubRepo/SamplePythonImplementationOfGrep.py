# shell2.py

import os
from glob import glob
import numpy as np
"""Volume 3: Unix Shell 2.
Drew Henrichsen
Volume 3, Modelling
September 14 (due date)
"""


# Problem 5
def grep(target_string, file_pattern):
    """Finds all files in the current directory or its subdirectories that
    match the file pattern, then determines which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.
    """
    listOfCompatibleFiles = glob("**/" + file_pattern,recursive = True)
    for file in listOfCompatibleFiles:
        with open(file,'r') as f:
            file_text = f.read()
            if target_string in file_text:
                print(file)
                continue
    return
            
            
    

# Problem 6
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    toSearch = glob("**/*.*",recursive = True)
    fileSizes = []
    for file in toSearch:
        fileSizes.append(os.path.getsize(file))
    print(fileSizes)
    correctIndices = np.argsort(fileSizes)
    print(correctIndices)
    print(fileSizes[correctIndices[0]])
    to_return = [fileSizes[n] for n in correctIndices]
    print(to_return)
    finalArray = np.flip(correctIndices[-10:],axis =0)
    for n in finalArray:
        print(toSearch[n])
    
        
