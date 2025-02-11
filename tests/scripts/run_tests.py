import unittest
import yaml
import sys
import os
import pandas as pd
from snakemake.io import expand

class RapidsTests(unittest.TestCase):
    unittest.TestLoader.sortTestMethodsUsing = lambda self, a, b: (a < b) - (a > b)

    def generate_sensor_file_lists(self):
    # Go through the configs and select those sensors with COMPUTE = True.
    # Also get TIME_SEGMENTS, then create expected 
    # files. Return dictionary with list of file paths of expected and 
    # actual files for each sensor listed in the config file. Added for Travis.

    # Initialize string of file path for both expected and actual metric values
        segment = self.configs['TIME_SEGMENTS']['TYPE'].lower()
        act_str = "data/processed/features/{pid}/{sensor_key}.csv"
        exp_str = "tests/data/processed/features/"+segment+"/{pid}/{sensor_key}.csv"

        # Build list of sensors to be tested. 
        sensors = []
        for sensor in self.configs:
            if "PROVIDERS" in self.configs[sensor] and self.configs[sensor]["PROVIDERS"] is not None:
                for provider in self.configs[sensor]["PROVIDERS"]:
                    if self.configs[sensor]["PROVIDERS"][provider]["COMPUTE"]:
                        sensors.append(sensor.lower())

        act_file_list = expand(act_str,pid=self.configs["PIDS"],sensor_key = sensors)                                
        exp_file_list = expand(exp_str, pid=self.configs["PIDS"],sensor_key = sensors)
        sensor_file_lists = list(zip(act_file_list,exp_file_list))          

        return sensor_file_lists

    def test_sensors_files_exist(self):
        # Loop through the file_list dictionary and check if the files exist. 
        file_lists = self.generate_sensor_file_lists()
        for each in file_lists:
            print("The actual output file should exist: {}".format(each[0]))
            self.assertEqual(os.path.exists( each[0]), 1)


    def test_sensors_features_calculations(self):
        sensor_file_list = self.generate_sensor_file_lists()        
        for act_result, exp_result in sensor_file_list:
            df_act = pd.read_csv(act_result)
            df_exp = pd.read_csv(exp_result)
            if df_act.empty:
                print(act_result)
                print("The expected output should be empty: {}".format(exp_result))
                self.assertTrue(df_exp.empty)
            else:
                # The order in which the columns varies from time to time so
                # the columns are sorted before doing the comparision 
                print("Comparing: {} and {}".format(act_result, exp_result))
                df_exp = df_exp.reindex(sorted(df_exp.columns), axis=1)
                df_act = df_act.reindex(sorted(df_act.columns), axis=1)
                pd.testing.assert_frame_equal(df_exp, df_act, obj=df_exp)


class TestFrequency(RapidsTests):
    @classmethod
    def setUpClass(cls):
        # Runs once to Setup env
        # global configs 
        with open(r'tests/settings/frequency_config.yaml') as file:
            cls.configs = yaml.full_load(file)


class TestPeriodic(RapidsTests):
    @classmethod
    def setUpClass(cls):
        # Runs once to Setup env
        # global configs 
        with open(r'tests/settings/periodic_config.yaml') as file:
            cls.configs = yaml.full_load(file)


def run_some_tests(test_type):
    # Run only the tests in the specified classes
    if test_type == "frequency":
        test_class = TestFrequency
    elif test_type == "periodic":
        test_class = TestPeriodic
    else:
        raise ValueError("Only frequency or periodic are valid arguments")
    loader = unittest.TestLoader()

    suite = loader.loadTestsFromTestCase(test_class)
    big_suite = unittest.TestSuite(suite)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)


if __name__ == '__main__':
    run_some_tests(sys.argv[1])