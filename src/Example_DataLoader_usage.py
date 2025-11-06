from src.DataLoader import DataLoader
# creates a DataLoader class object that creates a structure described in the docs folder (data_structure_spec.md) from raw data
data = DataLoader("W_010",False)
data.set_EEG_data("../DATA/W_010/")
data.save_to_file()
# usage of staticmethod load_output_data loads data created by DataLoader
out = DataLoader.load_output_data("/Users/admin/PycharmProjects/hyperscanning-signal-analysis/DATA/OUT/W_010.joblib")