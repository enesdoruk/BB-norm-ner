from os import listdir, write
from os.path import isfile, join, splitext
import copy
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.mention = X
        self.label = y

    def __getitem__(self, idx):
        mention = self.mention[idx]
        label = self.label[idx]
        sample = (mention, label)
        return sample

    def __len__(self):
        return len(self.mention)
    
    

def loader_one_bb4_fold(l_repPath):
    """
    Description: Load BB4 data from files.
    WARNING: OK only if A1 file is read before its A2 file (normally the case).
    :param l_repPath: A list of directory path containing set of A1 (and possibly A2) files.
    :return:
    """
    ddd_data = dict()

    i = 0
    for repPath in l_repPath:

        for fileName in listdir(repPath):
            filePath = join(repPath, fileName)

            if isfile(filePath):

                fileNameWithoutExt, ext = splitext(fileName)
                
                if ext == ".a1":

                    with open(filePath, encoding="utf8") as file:
                    
                        if fileNameWithoutExt not in ddd_data.keys():

                            ddd_data[fileNameWithoutExt] = dict()
                            for line in file:

                                l_line = line.split('\t')

                                if l_line[1].split(' ')[0] == "Title" or l_line[1].split(' ')[0] == "Paragraph":
                                    pass
                                else:
                                    exampleId = "bb4_" + "{number:06}".format(number=i)

                                    ddd_data[fileNameWithoutExt][exampleId] = dict()

                                    ddd_data[fileNameWithoutExt][exampleId]["T"] = l_line[0]
                                    ddd_data[fileNameWithoutExt][exampleId]["type"] = l_line[1].split(' ')[0]
                                    ddd_data[fileNameWithoutExt][exampleId]["mention"] = l_line[2].rstrip()
                                    if "cui" not in ddd_data[fileNameWithoutExt][exampleId].keys():
                                        ddd_data[fileNameWithoutExt][exampleId]["cui"] = list()
                                    i += 1

        for fileName in listdir(repPath):
            filePath = join(repPath, fileName)

            if isfile(filePath):

                fileNameWithoutExt, ext = splitext(fileName)
                
                if ext == ".a2":

                    with open(filePath, encoding="utf8") as file:

                        if fileNameWithoutExt in ddd_data.keys():

                            for line in file:
                                l_line = line.split('\t')

                                l_info = l_line[1].split(' ')
                                Tvalue = l_info[1].split(':')[1]

                                for id in ddd_data[fileNameWithoutExt].keys():
                                    if ddd_data[fileNameWithoutExt][id]["T"] == Tvalue:
                                        if ddd_data[fileNameWithoutExt][id]["type"] == "Habitat" or \
                                                ddd_data[fileNameWithoutExt][id]["type"] == "Phenotype" or \
                                                    ddd_data[fileNameWithoutExt][id]["type"] == "Microorganism":
                                            cui = "OBT:" + l_info[2].split('Referent:')[1].rstrip().replace('OBT:', '')
                                            ddd_data[fileNameWithoutExt][id]["cui"].append(cui)
                                        elif ddd_data[fileNameWithoutExt][id]["type"] == "Microorganism":
                                            cui = l_info[2].split('Referent:')[1].rstrip()
                                            ddd_data[fileNameWithoutExt][id]["cui"] = [cui]  # No multi-normalization for microorganisms
    return ddd_data


def extract_data(ddd_data, l_type=[]):
    """

    :param ddd_data:
    :param l_type:
    :return:
    """
    dd_data = dict()

    for fileName in ddd_data.keys():
        for id in ddd_data[fileName].keys():
            if ddd_data[fileName][id]["type"] in l_type:
                dd_data[id] = copy.deepcopy(ddd_data[fileName][id])
    return dd_data