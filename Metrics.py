import pandas as pd


class Metrics(object):

    def __init__(self, target, result):
        self.target = target
        self.result = result

    def efficiency(self):
        if type(self.target) != type(self.result):
            print("Dtypes of target and result does not match")
            return 0
        elif len(self.target) != len(self.result):
            print("Lengths of target and result does not match")
            return 0

        count = 0
        for i in range(len(self.target)):
            if self.target[i] == self.result[i]:
                count += 1
        return count / len(self.target) * 100

    def confusion_matrix(self):
        if type(self.target) != type(self.result):
            print("Dtypes of target and result does not match")
            return pd.DataFrame
        elif len(self.target) != len(self.result):
            print("Lengths of target and result does not match")
            return pd.DataFrame

        conf_mtrx = {'Positives': [0, 0],
                     'Negatives': [0, 0]}

        for i in range(len(self.target)):
            if self.result[i] == 1:
                if self.result[i] == self.target[i]:
                    conf_mtrx['Positives'][0] += 1
                else:
                    conf_mtrx['Positives'][1] += 1
            else:
                if self.result[i] == self.target[i]:
                    conf_mtrx['Negatives'][0] += 1
                else:
                    conf_mtrx['Negatives'][1] += 1
        return pd.DataFrame(conf_mtrx, index=['True', 'False'])

