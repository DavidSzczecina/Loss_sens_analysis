import numpy as np 
from numpy import ndarray
from typing import Tuple, List

def compute(confusion_matrix: ndarray, num_classes: int) -> Tuple[float, float]:
    #Iterate through each correct class 
    sensitivity_arr: List[float] = []
    specificity_arr: List[float] = []

    for i in range(num_classes): 
        #Value on the diagonal for true positive, to calculate true negative 
        TP: int = confusion_matrix[i,i]
        #Sums the i'th row, subtracting the sole true positive, to calculate true negative 
        FN: int = np.sum(confusion_matrix[i, :]) - TP
        #Sums the i'th column subtracting the sole true positive, to calculate false positive 
        FP: int = np.sum(confusion_matrix[:, i]) - TP
        #The sum of the matrix, less the other components 
        TN: int = np.sum(confusion_matrix) - TP - FN - FP
        #Calculate the sensitivity and specificity for the class
        sensitivity: float = float(TP)/(TP + FN)
        specificity: float = float(TN)/(TN + FP)
        #Add values to the arrays
        sensitivity_arr.append(sensitivity)
        specificity_arr.append(specificity)

    #Average all sensitivities and specificities accross all classes
    avg_sens: float = round(np.sum(sensitivity_arr)/num_classes,4)*100
    avg_spec: float = round(np.sum(specificity_arr)/num_classes,4)*100

    #Return the values 
    return avg_sens, avg_spec