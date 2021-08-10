
import os

class Settings:
    def __init__(self):
        current_folder = os.path.dirname(__file__) + '/'
        with open(current_folder + 'settings.txt', 'r') as f:
            for line in f:
                line = line.replace(' ', '').replace('\n', '')
                splits = line.split(':')
                if splits[0] == 'delimiter':
                    self.delimiter = line[line.index(':') + 1:]
                elif splits[0] == 'end_of_file':
                    self.end_of_file = splits[1]
                elif splits[0] == 'global_setting_keyword':
                    self.global_setting_keyword = splits[1]
                elif splits[0] == 'networks_file':
                    self.networks_file = current_folder + splits[1]
                elif splits[0] == 'batch_sizes_file':
                    self.batch_sizes_file = current_folder + splits[1]
                elif splits[0] == 'metrics_file':
                    self.metrics_file = current_folder + 'out/' + splits[1]
                elif splits[0] == 'input_dims_file':
                    self.input_dims_file = current_folder + splits[1]
                elif splits[0] == 'precisions_file':
                    self.precisions_file = current_folder + splits[1]
                elif splits[0] == 'tflite_folder':
                    self.tflite_folder = current_folder + splits[1]
