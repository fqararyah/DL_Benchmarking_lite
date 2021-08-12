import os
import datetime

import tflite_runtime.interpreter as tflite

import Settings
import pathlib

import numpy as np
#from tensorflow.keras import datasets
import time

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

test_images = []

def load_images():
    global test_images
    if len(test_images) == 0:
        (train_images, train_labels), (test_images,
                                        test_labels) = datasets.cifar10.load_data()
    return test_images

class BenchmarkModel:
    def __init__(self, model_name = '', batch_sizes = [1], inputs_dims = [[32, 32]], bit_widths = []):
        self.model_name = model_name
        self.batch_sizes = batch_sizes
        self.inputs_dims = inputs_dims
        self.bit_widths = bit_widths

    def get_metrics(self):
        #load_images()
        for input_dim in self.inputs_dims:
            test_images = np.random.randint(low =0, high= 256, size = [1000, input_dim[0], input_dim[1],\
                 3], dtype=np.uint8)
            test_images_preprocessed = test_images / 255.0
            test_images_preprocessed = test_images_preprocessed[0:max(self.batch_sizes) * 10]
            for bit_width in self.bit_widths:
                tflite_models_dir = pathlib.Path(Settings.Settings().tflite_folder)
                tflite_models_dir.mkdir(exist_ok=True, parents=True)
                if bit_width == 32:
                    tflite_model_file = tflite_models_dir/(self.model_name+"model_quant_32.tflite")
                elif bit_width == 16:
                    tflite_model_file = tflite_models_dir/(self.model_name+"model_quant_16.tflite")
                interpreter = tflite.Interpreter(model_path=str(tflite_model_file), experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

                self.get_metrics_quantized(input_dim, test_images_preprocessed, test_images, bit_width, interpreter)


    def get_metrics_32(self, input_dim, test_images_preprocessed, test_images):
        with open(Settings.Settings().metrics_file + '_' +self.model_name + '_32_' + str(input_dim[0]) + 'x' + \
            str(input_dim[1]) + '_' + str(datetime.datetime.now()).split('.')[0], 'w') as f:
            for batch_size in self.batch_sizes:
                f.write('\n----------------\nbatch size: ' + str(batch_size) + '\n----------------\n')
                #this is to load the model
                """ tmp = np.argmax(self.pretrained_model.predict(x = test_images[max(self.batch_sizes)*10:max(self.batch_sizes)*20]/255.0, batch_size = \
                    batch_size, verbose = 0))
                #Throughput
                t0 = time.time()
                #test_loss, test_acc = pretrained_model.evaluate(test_images,  test_labels)#, verbose=2)
                tmp = np.argmax(self.pretrained_model.predict(x = test_images_preprocessed, batch_size = batch_size, verbose = 0), 1)
                f.write("Execution time is: " + str((time.time() - t0) / len(test_images_preprocessed)) + "seconds.\n") """
                #end throughput

                #latency
                avg_time = 0.0
                #avg_time_with_preprocessing = 0.0
                counter = 0
                while counter * batch_size < max(self.batch_sizes) * 10:
                    image_batch = test_images[counter * batch_size: (counter + 1) * batch_size]
                    #t0_with_preprocessing = time.time()
                    tmp = -1
                    t0 = time.time()
                    image_batch = image_batch / 255.0
                    tmp = np.argmax(self.pretrained_model(image_batch, training = False))
                    if tmp != -1 and counter > 0:
                        avg_time += time.time() - t0
                    #avg_time_with_preprocessing += time.time() - t0_with_preprocessing
                    counter += 1

                avg_time /= (len(test_images_preprocessed) -1)
                #avg_time_with_preprocessing /= counter
                f.write("Latency is: " + str(avg_time) + " seconds.\n")
                #f.write("Latency (with processing time) is: " + str(avg_time_with_preprocessing) + " seconds.\n")
                #end latency

    def get_metrics_quantized(self, input_dim, test_images_preprocessed, test_images, no_of_bits, interpreter):
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        with open(Settings.Settings().metrics_file + '_' +self.model_name + '_' + str(no_of_bits) + '_' + \
            str(input_dim[0]) + 'x' + str(input_dim[1]) + '_' + str(datetime.datetime.now()).split('.')[0], 'w') \
                as f:
            for batch_size in self.batch_sizes:
                f.write('\n----------------\nbatch size: ' + str(batch_size) + '\n----------------\n')
                #latency
                avg_latency = 0
                avg_time = 0
                #avg_time_with_preprocessing = 0.0
                counter = 0
                interpreter.resize_tensor_input(0,[batch_size, input_dim[0], input_dim[1], 3])
                interpreter.allocate_tensors()
                while counter * batch_size < max(self.batch_sizes) * 10:
                    image_batch = test_images[counter * batch_size: (counter + 1) * batch_size]
                    t0 = time.time()
                    image_batch = image_batch.astype(np.float32)
                    image_batch = image_batch / 255.0
                    interpreter.set_tensor(input_index, image_batch)
                    t1 = time.time()
                    interpreter.invoke()
                    predictions = interpreter.get_tensor(output_index)
                    predicted = np.argmax(predictions, 0)
                    if len(predicted) > 0 and counter > 0:
                        avg_time += time.time() - t1
                        avg_latency += time.time() - t0
                    #avg_time_with_preprocessing += time.time() - t0_with_preprocessing
                    counter += 1
                avg_time /= (len(test_images_preprocessed) - 1)
                avg_latency /= (len(test_images_preprocessed) - 1)
                #avg_time_with_preprocessing /= counter
                f.write("Execution time is: " + str(avg_time) + " seconds.\n")
                f.write("Latency is: " + str(avg_latency) + " seconds.\n")
                #f.write("Latency (with processing time) is: " + str(avg_time_with_preprocessing) + " seconds.\n")
                #end latency
