import numpy as np
from typing import List, Union, Set
import torch
import torch.nn as nn
import os
import pandas as pd
import json
import csv


# Extract a dataframe given a root directory and filename. This function will look in subdirectories, returns None if not found
def getDataframe(rootDir: str, fileName: str) -> Union[None, pd.DataFrame]:
    fullPath: str = ""
    isFound: bool = False
    dataframe: pd.DataFrame
    for dirpath, _, filenames in os.walk(rootDir):
        if fileName in filenames:
            fullPath = os.path.join(dirpath, fileName)
            isFound = True
            break

    if isFound:
        dataframe = pd.read_csv(fullPath)
        return dataframe
    else:
        return None


def getJobMetadata(rootDir: str) -> Union[dict, None]:
    fullPath: str = ""
    isFound: bool = False
    jobJson: dict
    for dirpath, _, filenames in os.walk(rootDir):
        if "job_details.json" in filenames:
            fullPath = os.path.join(dirpath, "job_details.json")
            isFound = True
            break
    if isFound:
        with open(fullPath, "r") as file:
            jobJson = json.load(file)
            return jobJson
    else:
        return None


# Gets all files_x jobs, returns list of job directories, not full path
def getAllJobs(rootDir: str) -> List[str]:
    allItems: List[str] = os.listdir(rootDir)
    allJobs: List[str] = []
    for item in allItems:
        itemPath: str = os.path.join(rootDir, item)

        if os.path.isdir(itemPath) and "files_" in item:
            allJobs.append(item)

    return allJobs


def setVersion(jobs: List[str], thisJob: str) -> None:
    thisJobJsonPath: str = os.path.join("../job_outputs", thisJob, "job_details.json")
    thisJobJson: dict = {}
    num_version: int = 1
    with open(thisJobJsonPath, "r") as file:
        thisJobJson = json.load(file)

    thisJobModel: str = thisJobJson.get("model_name", "error")
    thisJobHyperparams: dict = thisJobJson.get("hyperparams")
    thisJobCriterion: str = thisJobJson.get("criterion")
    thisJobOptimizer: str = thisJobJson.get("optimizer")
    thisJobScheduler: str = thisJobJson.get("scheduler")
    seenVersions: Set[int] = set()
    # For each job in all jobs, compare the job model and if they match, either increment version or set version
    for job in jobs:
        relativeJsonPath: str = os.path.join("../job_outputs", job, "job_details.json")
        tempJobJson: dict
        tempJobModel: str
        try:
            with open(relativeJsonPath, "r") as file:
                tempJobJson = json.load(file)
                tempJobModel: str = tempJobJson.get("model_name", "error")
                tempJobHyperparams: dict = tempJobJson.get("hyperparams")
                tempJobCriterion: str = tempJobJson.get("criterion")
                tempJobOptimizer: str = tempJobJson.get("optimizer")
                tempJobScheduler: str = tempJobJson.get("scheduler")

            if tempJobModel == thisJobModel and "version" in tempJobJson:
                if (
                    tempJobHyperparams == thisJobHyperparams
                    and tempJobCriterion == thisJobCriterion
                    and tempJobOptimizer == thisJobOptimizer
                    and tempJobScheduler == thisJobScheduler
                ):
                    num_version = tempJobJson.get("version", "error")
                    break
                else:
                    tempVersion: int = tempJobJson.get("version", "error")
                    if tempVersion not in seenVersions:
                        num_version += 1
                        seenVersions.add(tempVersion)
        except:
            print("No job_details.json path found for ", relativeJsonPath)

    thisJobJson["version"] = num_version

    if thisJobModel == "error" or tempJobModel == "error" or num_version == "error":
        print("thisJobModel", thisJobModel)
        print("tempJobModel", tempJobModel)
        print("num_version", num_version)
        raise ValueError("error occurred")

    with open(thisJobJsonPath, "w") as file:
        json.dump(thisJobJson, file, indent=4)


def equalDicts(dict1: dict, dict2: dict, ignore_keys: list) -> bool:
    # Create copies of the dictionaries
    copy1 = dict1.copy()
    copy2 = dict2.copy()

    # Remove ignored fields
    for key in ignore_keys:
        copy1.pop(key, None)  # The `None` ensures no error if the key is not found
        copy2.pop(key, None)

    # Compare the modified dictionaries
    return copy1 == copy2


# Returns job directories of matching jobs based on their job_details.json file files
def getMatchingJobs(trueMetadata: dict, jobs: List[str]) -> List[str]:
    matchingJobs: List[str] = []
    ignore_keys = ["creation_date"]
    for job in jobs:
        # All functions assume gt_sens.sh is called from slurm_scripts directory
        relativeJobPath: str = os.path.join("../job_outputs", job)
        tempMetadata: dict = getJobMetadata(relativeJobPath)
        if tempMetadata == None:
            continue
        # Checks if the *this job's metadata is equal to the passed in jobs metadatas
        if equalDicts(trueMetadata, tempMetadata, ignore_keys):
            matchingJobs.append(job)
    return matchingJobs


# Compute summary stats and write to it to the job file
def computeSummaryStats(
    matchingJobs: List[str], thisJob: str, metric: str
) -> pd.DataFrame:
    thisMetadataPath = os.path.join("../job_outputs", thisJob)
    thisJobMetadata: dict = getJobMetadata(thisMetadataPath)
    thisEpoch: int = thisJobMetadata.get("num_epochs")
    thisCorruptionRateList: List[int] = thisJobMetadata.get("corruption_arr")
    thisCorruptionRateSeries: pd.Series = pd.Series(thisCorruptionRateList)
    thisJobCsv: str = os.path.join(
        "../job_outputs", thisJob, "csv_data/summary_stats_" + metric + ".csv"
    )
    summaryDataframe: pd.DataFrame = pd.DataFrame()
    summaryDataframe["corruption_rate"] = thisCorruptionRateSeries

    #  iterate through matching jobs, find the job directory, get the data.csv, filter for the final epoch,
    #  get series for the specified metric, add that series to the sumaryDataframe
    for job in matchingJobs:
        relativeJobPath: str = os.path.join("../job_outputs", job)
        tempDataframe: pd.DataFrame = getDataframe(relativeJobPath, "data.csv")
        finalEpochDf: pd.DataFrame = tempDataframe[tempDataframe["epoch"] == thisEpoch]
        jobMetricSeries: pd.Series = finalEpochDf[metric]
        # Index are based on their original index, ie after filtering epochs, index remains the same (2, 5, 8) instead of (0, 1, 2)
        # Need to reset the indexes to insert the column correctly
        jobMetricSeries = jobMetricSeries.reset_index(drop=True)
        summaryDataframe[job] = jobMetricSeries

    # Start and end column to iterate through
    firstFile: str = matchingJobs[0]
    lastFile: str = matchingJobs[len(matchingJobs) - 1]

    medianList: List[float] = []
    meanList: List[float] = []
    stDevList: List[float] = []
    varianceList: List[float] = []
    rangeList: List[float] = []
    q25List: List[float] = []
    q75List: List[float] = []
    minList: List[float] = []
    maxList: List[float] = []
    IQRList: List[float] = []

    # Iterate row-wise, getting the values in that row (corruption rate) for all jobs
    for index, row in summaryDataframe.iterrows():
        row_values: List[float] = row.loc[firstFile:lastFile]
        medianList.append(np.median(row_values))
        meanList.append(np.mean(row_values))
        stDevList.append(np.std(row_values))
        varianceList.append(np.var(row_values))
        q25List.append(np.percentile(row_values, 25))
        q75List.append(np.percentile(row_values, 75))
        minList.append(np.min(row_values))
        maxList.append(np.max(row_values))
        IQRList.append(np.percentile(row_values, 75) - np.percentile(row_values, 25))
        rangeList.append(np.max(row_values) - np.min(row_values))

    summaryDataframe["mean"] = meanList
    summaryDataframe["median"] = medianList
    summaryDataframe["stdev"] = stDevList
    summaryDataframe["variance"] = varianceList
    summaryDataframe["range"] = rangeList
    summaryDataframe["q25"] = q25List
    summaryDataframe["q75"] = q75List
    summaryDataframe["min"] = minList
    summaryDataframe["max"] = maxList
    summaryDataframe["IQR"] = IQRList

    # write to the csv file
    summaryDataframe.to_csv(thisJobCsv, index=False)

    return summaryDataframe


def computeRobustness(thisJob: str):
    relativeJobPath: str = os.path.join("../job_outputs", thisJob)
    thisJobMetadata: dict = getJobMetadata(relativeJobPath)
    corruptionRateList: List[int] = thisJobMetadata.get("corruption_arr")
    thisJobDataFrame: pd.DataFrame = getDataframe(relativeJobPath, "data.csv")
    finalEpoch: int = thisJobMetadata.get("num_epochs")
    finalEpochDf: pd.DataFrame = thisJobDataFrame[
        thisJobDataFrame["epoch"] == finalEpoch
    ]
    accuracySeries: pd.Series = finalEpochDf.get("accuracy")
    accuracySeries = accuracySeries.reset_index(drop=True)

    zeroCorruptionAcc: int = accuracySeries[0]

    thisJobMetadata["modelRobustness"] = 100

    for i, acc in enumerate(accuracySeries):
        if abs(zeroCorruptionAcc - acc) > 5:
            prevRobustness = float(corruptionRateList[i - 1])
            prevAccuracy = float(accuracySeries[i - 1])
            robustness = float(corruptionRateList[i])
            accuracy = float(accuracySeries[i])
            interpolatedRobustness = prevRobustness + abs(
                zeroCorruptionAcc - 5 - prevAccuracy
            ) * ((robustness - prevRobustness) / (accuracy - prevAccuracy))
            # x + y * (delta x / delta y)
            thisJobMetadata["modelRobustness"] = (
                interpolatedRobustness * 100
            )  # convert from decimal to percent

            # Write the the job details to a json for use in versioning
            with open(f"{relativeJobPath}/job_details.json", "w") as file:
                json.dump(thisJobMetadata, file, indent=4)
            break
