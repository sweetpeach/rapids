import pandas as pd
from datetime import datetime
import itertools

def featuresFullNames(intraday_features_to_compute, sleep_levels_to_compute, sleep_types_to_compute, consider_all):
    
    features_fullname = ["local_segment"]

    sleep_level_with_group = []
    for sleep_level_group in sleep_levels_to_compute:
        for sleep_level in sleep_levels_to_compute[sleep_level_group]:
            sleep_level_with_group.append(sleep_level + sleep_level_group.lower())
    
    if consider_all:
        features_fullname.extend([x[0] + x[1] + x[2] for x in itertools.product(intraday_features_to_compute["LEVELS_AND_TYPES"], sleep_level_with_group + ["all"], sleep_types_to_compute + ["all"])])
    else:
        features_fullname.extend([x[0] + x[1] + x[2] for x in itertools.product(intraday_features_to_compute["LEVELS_AND_TYPES"], sleep_level_with_group, sleep_types_to_compute)])
    if "ACROSS_LEVELS" in intraday_features_to_compute["RATIOS_SCOPE"]:
        features_fullname.extend(["ratio" + x[0] + x[1] for x in itertools.product(intraday_features_to_compute["RATIOS_TYPE"], sleep_level_with_group)])
    if "ACROSS_TYPES" in intraday_features_to_compute["RATIOS_SCOPE"] and "main" in sleep_types_to_compute:
        features_fullname.extend(["ratio" + x + "main" for x in intraday_features_to_compute["RATIOS_TYPE"]])
    if "WITHIN_LEVELS" in intraday_features_to_compute["RATIOS_SCOPE"]:
        features_fullname.extend(["ratio" + x[0] + x[1] + "within" + x[2] for x in itertools.product(intraday_features_to_compute["RATIOS_TYPE"], sleep_types_to_compute, sleep_level_with_group)])
    if "WITHIN_TYPES" in intraday_features_to_compute["RATIOS_SCOPE"]:
        features_fullname.extend(["ratio" + x[0] + x[1] + "within" + x[2] for x in itertools.product(intraday_features_to_compute["RATIOS_TYPE"], sleep_level_with_group, sleep_types_to_compute)])
    features_fullname.extend(intraday_features_to_compute["ROUTINE"])
    return features_fullname

def mergeSleepEpisodes(sleep_data, cols_for_groupby):
    
    sleep_episodes = pd.DataFrame(columns=["local_segment", "duration", "start_timestamp", "end_timestamp", "local_start_date_time", "local_end_date_time"])

    if cols_for_groupby and (not sleep_data.empty):
        sleep_data = sleep_data.groupby(by=cols_for_groupby)
        sleep_episodes = sleep_data[["duration"]].sum()
        sleep_episodes["start_timestamp"] = sleep_data["start_timestamp"].first()
        sleep_episodes["end_timestamp"] = sleep_data["end_timestamp"].last()
        sleep_episodes["local_start_date_time"] = sleep_data["local_start_date_time"].first()
        sleep_episodes["local_end_date_time"] = sleep_data["local_end_date_time"].last()
    
        sleep_episodes.reset_index(inplace=True, drop=False)

    return sleep_episodes

def statsFeatures(sleep_episodes, features, episode_type):
    
    episode_features = pd.DataFrame(columns=[feature + episode_type for feature in features])
    if sleep_episodes.empty:
        return episode_features

    if "countepisode" in features:
        episode_features["countepisode" + episode_type] = sleep_episodes[["local_segment", "duration"]].groupby(["local_segment"])["duration"].count()
    if "sumduration" in features:
        episode_features["sumduration" + episode_type] = sleep_episodes[["local_segment", "duration"]].groupby(["local_segment"])["duration"].sum()
    if "maxduration" in features:
        episode_features["maxduration" + episode_type] = sleep_episodes[["local_segment", "duration"]].groupby(["local_segment"])["duration"].max()
    if "minduration" in features:
        episode_features["minduration" + episode_type] = sleep_episodes[["local_segment", "duration"]].groupby(["local_segment"])["duration"].min()
    if "avgduration" in features:
        episode_features["avgduration" + episode_type] = sleep_episodes[["local_segment", "duration"]].groupby(["local_segment"])["duration"].mean()
    if "medianduration" in features:
        episode_features["medianduration" + episode_type] = sleep_episodes[["local_segment", "duration"]].groupby(["local_segment"])["duration"].median()
    if "stdduration" in features:
        episode_features["stdduration" + episode_type] = sleep_episodes[["local_segment", "duration"]].groupby(["local_segment"])["duration"].std()
    
    return episode_features

def allStatsFeatures(sleep_data, base_sleep_levels, base_sleep_types, features, sleep_intraday_features):

    # For CLASSIC
    for sleep_level, sleep_type in itertools.product(base_sleep_levels["CLASSIC"] + ["all"], base_sleep_types + ["all"]):
        sleep_episodes_classic = sleep_data[sleep_data["type"] == "classic"]
        sleep_episodes_classic = sleep_episodes_classic[sleep_episodes_classic["is_main_sleep"] == (1 if sleep_type == "main" else 0)] if sleep_type != "all" else sleep_episodes_classic
        sleep_episodes_classic = sleep_episodes_classic[sleep_episodes_classic["level"] == sleep_level] if sleep_level != "all" else sleep_episodes_classic
        sleep_intraday_features = pd.concat([sleep_intraday_features, statsFeatures(sleep_episodes_classic, features, sleep_level + "classic" + sleep_type)], axis=1)
    
    # For STAGES
    for sleep_level, sleep_type in itertools.product(base_sleep_levels["STAGES"] + ["all"], base_sleep_types + ["all"]):
        sleep_episodes_stages = sleep_data[sleep_data["type"] == "stages"]
        sleep_episodes_stages = sleep_episodes_stages[sleep_episodes_stages["is_main_sleep"] == (1 if sleep_type == "main" else 0)] if sleep_type != "all" else sleep_episodes_stages
        sleep_episodes_stages = sleep_episodes_stages[sleep_episodes_stages["level"] == sleep_level] if sleep_level != "all" else sleep_episodes_stages
        sleep_intraday_features = pd.concat([sleep_intraday_features, statsFeatures(sleep_episodes_stages, features, sleep_level + "stages" + sleep_type)], axis=1)
    
    # For UNIFIED
    for sleep_level, sleep_type in itertools.product(base_sleep_levels["UNIFIED"] + ["all"], base_sleep_types + ["all"]):
        sleep_episodes_unified = sleep_data[sleep_data["is_main_sleep"] == (1 if sleep_type == "main" else 0)] if sleep_type != "all" else sleep_data
        sleep_episodes_unified = sleep_episodes_unified[sleep_episodes_unified["unified_level"] == (0 if sleep_level == "awake" else 1)] if sleep_level != "all" else sleep_episodes_unified
        sleep_episodes_unified = mergeSleepEpisodes(sleep_episodes_unified, ["local_segment", "unified_level_episode_id"]) 
        sleep_intraday_features = pd.concat([sleep_intraday_features, statsFeatures(sleep_episodes_unified, features, sleep_level + "unified" + sleep_type)], axis=1)
    
    # Ignore the levels (e.g. countepisode[all][main])
    for sleep_type in base_sleep_types + ["all"]:
        sleep_episodes_none = sleep_data[sleep_data["is_main_sleep"] == (1 if sleep_type == "main" else 0)] if sleep_type != "all" else sleep_data
        sleep_episodes_none = mergeSleepEpisodes(sleep_episodes_none, ["local_segment", "type_episode_id"])
        sleep_intraday_features = pd.concat([sleep_intraday_features, statsFeatures(sleep_episodes_none, features, "all" + sleep_type)], axis=1)

    return sleep_intraday_features


# Since all the stats features have been computed no matter they are requested or not,
# we can pick the related features to calculate the RATIOS features directly.
# Take ACROSS_LEVELS RATIOS features as an example:
# ratiocount[remstages] = countepisode[remstages][all] / countepisode[all][all]
def ratiosFeatures(sleep_intraday_features, ratios_types, ratios_scopes, sleep_levels, sleep_types):
    
    # Put sleep_level_group and sleep_level together.
    # For example:
    # input (sleep_levels): {"CLASSIC": ["awake", "restless", "asleep"], "UNIFIED": ["awake", "asleep"]}
    # output (sleep_level_with_group): [("classic", "awake"), ("classic", "restless"), ("classic", "asleep"), ("unified", "awake"), ("unified", "asleep")]
    sleep_level_with_group = []
    for sleep_level_group in sleep_levels:
        for sleep_level in sleep_levels[sleep_level_group]:
            sleep_level_with_group.append((sleep_level_group.lower(), sleep_level))

    # ACROSS LEVELS
    if "ACROSS_LEVELS" in ratios_scopes:
        # Get the cross product of ratios_types and sleep_level_with_group.
        # For example:
        # input: ratios_types is ["count", "duration"], sleep_level_with_group is [("classic", "awake"), ("classic", "restless"), ("unified", "asleep")]
        # output: 
        # 1) ratios_type: "count", sleep_levels_combined: ("classic", "awake")
        # 2) ratios_type: "count", sleep_levels_combined: ("classic", "restless")
        # 3) ratios_type: "count", sleep_levels_combined: ("unified", "asleep")
        # 4) ratios_type: "duration", sleep_levels_combined: ("classic", "awake")
        # 5) ratios_type: "duration", sleep_levels_combined: ("classic", "restless")
        # 6) ratios_type: "duration", sleep_levels_combined: ("unified", "asleep")
        for ratios_type, sleep_levels_combined in itertools.product(ratios_types, sleep_level_with_group):
            sleep_level_group, sleep_level = sleep_levels_combined[0], sleep_levels_combined[1]
            agg_func = "countepisode" if ratios_type == "count" else "sumduration"
            across_levels = (sleep_intraday_features[agg_func + sleep_level + sleep_level_group + "all"] / sleep_intraday_features[agg_func + "all" + sleep_level_group + "all"]).to_frame().rename(columns={0: "ratio" + ratios_type + sleep_level + sleep_level_group})
            sleep_intraday_features = pd.concat([sleep_intraday_features, across_levels], axis=1)
    
    # ACROSS TYPES
    if "ACROSS_TYPES" in ratios_scopes:
        for ratios_type in ratios_types:
            agg_func = "countepisode" if ratios_type == "count" else "sumduration"
            across_types = (sleep_intraday_features[agg_func + "allmain"] / sleep_intraday_features[agg_func + "allall"]).to_frame().rename(columns={0: "ratio" + ratios_type + "main"})
            sleep_intraday_features = pd.concat([sleep_intraday_features, across_types], axis=1)
    
    # Get the cross product of ratios_types, sleep_level_with_group, and sleep_types.
    # For example:
    # input:
    # ratios_types is ["count", "duration"]
    # sleep_level_with_group is [("classic", "awake"), ("unified", "asleep")]
    # sleep_types is ["main", "nap"]
    # output:
    # 1) ratios_type: "count", sleep_levels_combined: ("classic", "awake"), sleep_type: "main"
    # 2) ratios_type: "count", sleep_levels_combined: ("classic", "awake"), sleep_type: "nap"
    # 3) ratios_type: "count", sleep_levels_combined: ("unified", "asleep"), sleep_type: "main"
    # 4) ratios_type: "count", sleep_levels_combined: ("unified", "asleep"), sleep_type: "nap"
    # 5) ratios_type: "duration", sleep_levels_combined: ("classic", "awake"), sleep_type: "main"
    # 6) ratios_type: "duration", sleep_levels_combined: ("classic", "awake"), sleep_type: "nap"
    # 7) ratios_type: "duration", sleep_levels_combined: ("unified", "asleep"), sleep_type: "main"
    # 8) ratios_type: "duration", sleep_levels_combined: ("unified", "asleep"), sleep_type: "nap"    
    for ratios_type, sleep_levels_combined, sleep_type in itertools.product(ratios_types, sleep_level_with_group, sleep_types):
        sleep_level_group, sleep_level = sleep_levels_combined[0], sleep_levels_combined[1]
        agg_func = "countepisode" if ratios_type == "count" else "sumduration"

        # WITHIN LEVELS
        if "WITHIN_LEVELS" in ratios_scopes:
            within_levels = (sleep_intraday_features[agg_func + sleep_level + sleep_level_group + sleep_type] / sleep_intraday_features[agg_func + sleep_level + sleep_level_group + "all"]).to_frame().rename(columns={0: "ratio" + ratios_type + sleep_type + "within" + sleep_level + sleep_level_group})
            sleep_intraday_features = pd.concat([sleep_intraday_features, within_levels], axis=1)

        # WITHIN TYPES
        if "WITHIN_TYPES" in ratios_scopes:
            within_types = (sleep_intraday_features[agg_func + sleep_level + sleep_level_group + sleep_type] / sleep_intraday_features[agg_func + "all" + sleep_level_group + sleep_type]).to_frame().rename(columns={0: "ratio" + ratios_type + sleep_level + sleep_level_group + "within" + sleep_type})
            sleep_intraday_features = pd.concat([sleep_intraday_features, within_types], axis=1)

    return sleep_intraday_features


def singleSleepTypeRoutineFeatures(sleep_intraday_data, routine, routine_reference_time, sleep_type, sleep_intraday_features):

    sleep_intraday_data = sleep_intraday_data[sleep_intraday_data["is_main_sleep"] == (1 if sleep_type == "mainsleep" else 0)]
    if "starttimefirst" + sleep_type in routine:
        grouped_first = sleep_intraday_data.groupby(["local_segment"]).first()
        if routine_reference_time == "MIDNIGHT":
            sleep_intraday_features["starttimefirst" + sleep_type] = grouped_first["local_start_date_time"].apply(lambda x: x.hour * 60 + x.minute + x.second / 60)
        elif routine_reference_time == "START_OF_THE_SEGMENT":
            sleep_intraday_features["starttimefirst" + sleep_type] = (grouped_first["start_timestamp"] - grouped_first["segment_start_timestamp"]) / (60 * 1000)
        else:
            raise ValueError("Please check FITBIT_SLEEP_INTRADAY section of config.yaml: ROUTINE_REFERENCE_TIME can only be MIDNIGHT or START_OF_THE_SEGMENT.")
    
    if "endtimelast" + sleep_type in routine:
        grouped_last = sleep_intraday_data.groupby(["local_segment"]).last()
        if routine_reference_time == "MIDNIGHT":
            sleep_intraday_features["endtimelast" + sleep_type] = grouped_last["local_end_date_time"].apply(lambda x: x.hour * 60 + x.minute + x.second / 60)
        elif routine_reference_time == "START_OF_THE_SEGMENT":
            sleep_intraday_features["endtimelast" + sleep_type] = (grouped_last["end_timestamp"] - grouped_last["segment_start_timestamp"]) / (60 * 1000)
        else:
            raise ValueError("Please check FITBIT_SLEEP_INTRADAY section of config.yaml: ROUTINE_REFERENCE_TIME can only be MIDNIGHT or START_OF_THE_SEGMENT.")

    return sleep_intraday_features

def routineFeatures(sleep_intraday_data, routine, routine_reference_time, sleep_type, sleep_intraday_features):        
    
    if "starttimefirstmainsleep" in routine or "endtimelastmainsleep" in routine:
        sleep_intraday_features = singleSleepTypeRoutineFeatures(sleep_intraday_data, routine, routine_reference_time, "mainsleep", sleep_intraday_features)
    
    if "starttimefirstnap" in routine or "endtimelastnap" in routine:
        sleep_intraday_features = singleSleepTypeRoutineFeatures(sleep_intraday_data, routine, routine_reference_time, "nap", sleep_intraday_features)
    
    return sleep_intraday_features


def rapids_features(sensor_data_files, time_segment, provider, filter_data_by_segment, *args, **kwargs):

    sleep_intraday_data = pd.read_csv(sensor_data_files["sensor_data"])

    consider_all = provider["FEATURES"]["LEVELS_AND_TYPES_COMBINING_ALL"]
    include_sleep_later_than = provider["INCLUDE_SLEEP_LATER_THAN"]
    routine_reference_time = provider["ROUTINE_REFERENCE_TIME"]

    requested_intraday_features = provider["FEATURES"]
    requested_sleep_levels = provider["SLEEP_LEVELS"]
    requested_sleep_types = provider["SLEEP_TYPES"]

    # Name of the features this function can compute
    base_intraday_features = {"LEVELS_AND_TYPES": ["countepisode", "sumduration", "maxduration", "minduration", "avgduration", "medianduration", "stdduration"],
                                "RATIOS_TYPE": ["count", "duration"],
                                "RATIOS_SCOPE": ["ACROSS_LEVELS", "ACROSS_TYPES", "WITHIN_LEVELS", "WITHIN_TYPES"],
                                "ROUTINE": ["starttimefirstmainsleep", "endtimelastmainsleep", "starttimefirstnap", "endtimelastnap"]}
    base_sleep_levels = {"CLASSIC": ["awake", "restless", "asleep"],
                            "STAGES": ["wake", "deep", "light", "rem"],
                            "UNIFIED": ["awake", "asleep"]}
    base_sleep_types = ["main", "nap"]

    # The subset of requested features this function can compute
    intraday_features_to_compute = {key: list(set(requested_intraday_features[key]) & set(base_intraday_features[key])) for key in requested_intraday_features if key in base_intraday_features}
    sleep_levels_to_compute = {key: list(set(requested_sleep_levels[key]) & set(base_sleep_levels[key])) for key in requested_sleep_levels if key in base_sleep_levels}
    sleep_types_to_compute = list(set(requested_sleep_types) & set(base_sleep_types))

    # Full names
    features_fullnames = featuresFullNames(intraday_features_to_compute, sleep_levels_to_compute, sleep_types_to_compute, consider_all)
    sleep_intraday_features = pd.DataFrame(columns=features_fullnames)
    
    # Include sleep later than
    start_minutes = sleep_intraday_data.groupby("start_timestamp").first()["local_time"].apply(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]) + int(x.split(":")[2]) / 60).to_frame().rename(columns={"local_time": "start_minutes"}).reset_index()
    sleep_intraday_data = sleep_intraday_data.merge(start_minutes, on="start_timestamp", how="left")
    sleep_intraday_data = sleep_intraday_data[sleep_intraday_data["start_minutes"] >= include_sleep_later_than]
    del sleep_intraday_data["start_minutes"]

    sleep_intraday_data = filter_data_by_segment(sleep_intraday_data, time_segment)

    # While level_episode_id is based on levels provided by Fitbit (classic & stages), unified_level_episode_id is based on unified_level.
    sleep_intraday_data.insert(3, "unified_level_episode_id", (sleep_intraday_data[["type_episode_id", "unified_level"]] != sleep_intraday_data[["type_episode_id", "unified_level"]].shift()).any(axis=1).cumsum())

    if not sleep_intraday_data.empty:

        sleep_intraday_features = pd.DataFrame()

        # ALL LEVELS AND TYPES: compute all stats features no matter they are requested or not
        sleep_intraday_features = allStatsFeatures(sleep_intraday_data, base_sleep_levels, base_sleep_types, base_intraday_features["LEVELS_AND_TYPES"], sleep_intraday_features)

        # RATIOS: only compute requested features
        sleep_intraday_features = ratiosFeatures(sleep_intraday_features, intraday_features_to_compute["RATIOS_TYPE"], intraday_features_to_compute["RATIOS_SCOPE"], sleep_levels_to_compute, sleep_types_to_compute)
        
        # ROUTINE: only compute requested features
        sleep_intraday_features = routineFeatures(sleep_intraday_data, intraday_features_to_compute["ROUTINE"], routine_reference_time, sleep_types_to_compute, sleep_intraday_features)

        # Reset index and discard features which are not requested by user
        sleep_intraday_features.index.name = "local_segment"
        sleep_intraday_features.reset_index(inplace=True)
        sleep_intraday_features = sleep_intraday_features[features_fullnames]


    return sleep_intraday_features
