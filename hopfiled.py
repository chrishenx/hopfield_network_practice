import cv2, re, math, os, sys
import numpy as np

def load_pattern(image_path, image_size):
  image_path = (image_path).rstrip()
  print('Cargando:', image_path)
  image_pattern = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.intc, copy=False)
  print(image_pattern)
  (height, width) = image_pattern.shape
  if (height == image_size[0] and width == image_size[1]):
    image_pattern[image_pattern < 127] = -1.0
    image_pattern[image_pattern >= 127] = 1.0
    pattern = image_pattern.flatten()
    return pattern
          
"""
The images of the training set must be located at the training_set folder
and its paths in the patterns/image_patterns_paths.txt file.

@param image_size The allowed dimensions of the images to load
@return A list with the loaded images as binary

References:
  Binarize image: http://stackoverflow.com/questions/7624765/converting-an-opencv-image-to-black-and-white
""" 
def load_patterns(image_size):
  with open('patterns/image_patterns_paths.txt', 'r') as image_paths_file:
    accepted_extensions_pattern = re.compile(r'.*\.jpg|.*\.png') 
    patterns = []
    for image_path in image_paths_file:
      if (accepted_extensions_pattern.match(image_path) is not None):
        pattern = load_pattern('patterns/' + image_path, image_size)
        if pattern is not None:
          patterns.append(pattern)
    image_paths_file.close()
    return patterns

"""
Generates the weigthed matrix bases on a number of patterns

References:
  numpy efficient iteration: htttp://docs.scipy.org/doc/numpy/reference/arrays.nditer.html 
"""
def compute_weigthed_matrix(patterns, neuron_count):
  weigthed_matrix_range = neuron_count ** 2
  weigthed_matrix = np.zeros(weigthed_matrix_range ** 2).reshape(weigthed_matrix_range, weigthed_matrix_range)
  it = np.nditer(weigthed_matrix, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    (i, j) = (it.multi_index[0], it.multi_index[1])
    for pattern in patterns:
      it[0] += pattern[i] * pattern[j] / neuron_count
    it.iternext()
  return weigthed_matrix  
  
def parse_weigthed_matrix(file_name):
  with open(file_name, 'r') as file:
    try:
      weigthed_matrix = []
      last_row_len = None
      for line in file:
        row = list(map(float, line.split(',')))
        if last_row_len is not None and len(row) != last_row_len:
          return
        last_row_len = len(row)
        weigthed_matrix.append(row)
      if last_row_len != len(weigthed_matrix):
        return
      return np.array(weigthed_matrix)
    except ValueError:
    	 print('Error de parseo')
  
def transfer_function(array):
  array[array < 0] = -1 
  array[array >= 0] = 1
  return array
  
def associate(weigthed_matrix, pattern):
  inhibition_state = pattern
  done = False
  pattern_matched = True
  past_outputs = []
  last_output = transfer_function(np.array(inhibition_state))
  loop_number = 1
  while not done:
    inhibition_state = np.dot(weigthed_matrix, last_output)
    output = transfer_function(inhibition_state)
    if (last_output == output).all():
      done = True
      pattern_matched = True
    else:
      for past_output in past_outputs:
        if (past_outputs == output).all():
          print('Se ha detectado un ciclo')
          done = True
    print('Ciclo', loop_number)
    loop_number += 1
    last_output = output    
    past_outputs.append(last_output)
  return (pattern_matched, last_output)
     
if __name__ == '__main__':

  # Handling command line arguments
  
  parse_weigthed_matrix_from_file = False
  
  if len(sys.argv) == 2:
    parse_weigthed_matrix_from_file = sys.argv[1] == 'load_weigthed_matrix'
  
  NEURON_COUNT = 7  
  image_size = (NEURON_COUNT, NEURON_COUNT)
  
  weigthed_matrix = None

  patterns = load_patterns(image_size)
   
  if parse_weigthed_matrix_from_file:
    weigthed_matrix_file_path = 'patterns/weigthed_matrix.txt'
    print('Cargando matrix de pesos desde archivo "{0}"'.format('patterns/weigthed_matrix.txt'))
    weigthed_matrix = parse_weigthed_matrix(weigthed_matrix_file_path)
    if weigthed_matrix is None:
      weigthed_matrix = compute_weigthed_matrix(patterns, NEURON_COUNT * 1.0)
      if patterns is None or len(patterns) == 0:
        print('Error de usuario')
        exit(1)
    else:
      print('Matrix de pesos cargada de registro.')

  if weigthed_matrix is None:
    weigthed_matrix = compute_weigthed_matrix(patterns, NEURON_COUNT * 1.0)
    if patterns is None or len(patterns) == 0:
        print('Error de usuario')
        exit(1)
    file = open('patterns/weigthed_matrix.txt', 'w')
    for row in weigthed_matrix:
      file.write(','.join(map(str, row)) + '\n')
    file.close()
    
  pattern = load_pattern('patterns/diagonal.png', image_size)
  print(pattern)
  # print(pattern) # Useless line
 
  print('Tratando de asociar patron 1 ...')
  (pattern_matched, reconstructed_pattern) = associate(weigthed_matrix, pattern)
  if pattern_matched:
    generated_image = reconstructed_pattern.reshape(image_size)
    # TODO Evaluate if the following two lines are convinient
    generated_image[generated_image == -1] = 0
    generated_image[generated_image ==  1] = 255
    print(generated_image)
    print('Patron asociado')
    cv2.imshow('Patron recontruido', cv2.resize(generated_image, tuple([20 * x for x in image_size]), interpolation = cv2.INTER_NEAREST))
    cv2.waitKey(0)
  else:
    print('Patron no asociado') 
