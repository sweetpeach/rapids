import pandas as pd
from accelerometer.accelerometer_base import base_accelerometer_features
from accelerometer.arkit.ARKit import ARKit


acc_data = pd.read_csv(snakemake.input[0], parse_dates=["local_date_time", "local_date"])
day_segment = snakemake.params["day_segment"]
requested_acc_features = snakemake.params["features"]

arkit_params = snakemake.params["arkit_params"]
acc_features = pd.DataFrame(columns=["local_date"])
acc_features = acc_features.merge(base_accelerometer_features(acc_data, day_segment, requested_acc_features), on="local_date", how="outer")

arkit = ARKit(acc_data, day_segment, arkit_params)
arkit_features = arkit.run()

assert len(requested_acc_features) + 1 == acc_features.shape[1], "The number of features in the output dataframe (=" + str(acc_features.shape[1]) + ") does not match the expected value (=" + str(len(requested_features)) + " + 1). Verify your accelerometer feature extraction functions"

acc_features.to_csv(snakemake.output['base'], index=False)
arkit_features.to_csv(snakemake.output['arkit_features'], index=False)
