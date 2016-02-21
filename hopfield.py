import cv2, re, math, os, sys
import numpy as np

"""
Loads an image from image_path and returns its flatten version, but only if its dimensions are image_size
@param image_path The absolute or relative path of the image to load
@param image_size The dimentions that the loaded image must have
@return The flatten version of the image 
"""
def load_pattern(image_path, image_size):
  image_path = (image_path).rstrip()
  print('Cargando:', image_path)
  image_pattern = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.intc, copy=False)
  (height, width) = image_pattern.shape
  if (height == image_size[0] and width == image_size[1]):
    image_pattern[image_pattern < 127] = -1
    image_pattern[image_pattern >= 127] = 1
    pattern = image_pattern.flatten()
    return pattern
          
"""
The images of the training set must be located at the training_set folder
and its paths in the patterns/image_patterns_paths.txt file.

@param image_size The allowed dimensions of the images to load
@return A list with the loaded images as binary
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
        else:
          print('  Dimensiones invalidas.')
    image_paths_file.close()
    return patterns

"""
Generates the weigthed matrix bases on a number of patterns

References:
  numpy efficient iteration: htttp://docs.scipy.org/doc/numpy/reference/arrays.nditer.html 
"""
def compute_weighted_matrix(patterns, neuron_count):
  weighted_matrix = np.zeros(neuron_count ** 2).reshape(neuron_count, neuron_count)
  it = np.nditer(weighted_matrix, flags=['multi_index'], op_flags=['readwrite'], order='C')
  while not it.finished:
    (i, j) = (it.multi_index[0], it.multi_index[1])
    if i != j:
      for pattern in patterns:
        it[0] += pattern[i] * pattern[j] / neuron_count
    it.iternext()
  return weighted_matrix  
  
def parse_weighted_matrix(file_name):
  with open(file_name, 'r') as file:
    try:
      weighted_matrix = []
      last_row_len = None
      for line in file:
        row = list(map(float, line.split(',')))
        if last_row_len is not None and len(row) != last_row_len:
          return
        last_row_len = len(row)
        weighted_matrix.append(row)
      if last_row_len != len(weighted_matrix):
        return
      return np.array(weighted_matrix)
    except ValueError:
    	 print('Error de parseo')
  
def transfer_function(array):
  array[array < 0] = -1.0
  array[array >= 0] = 1.0
  return array
  
def associate(weighted_matrix, pattern):
  inhibition_state = pattern
  done = False
  pattern_matched = True
  past_outputs = []
  last_output = transfer_function(inhibition_state.astype(float, copy=False))
  iteration = 0
  process_file = open('patterns/last_association_process.csv', 'w')
  process_file.write('U({0}) = ,{1}\n'.format(iteration, ','.join(map(str, inhibition_state)) + '\n'))
  process_file.write('Y({0}) = ,{1}\n\n'.format(iteration, ','.join(map(str, last_output)) + '\n'))
  while not done:
    iteration += 1
    inhibition_state = np.dot(weighted_matrix, last_output)
    process_file.write('U({0}) = ,{1}\n'.format(iteration, ','.join(map(str, inhibition_state)) + '\n'))
    output = transfer_function(inhibition_state)
    process_file.write('Y({0}) = ,{1}\n\n'.format(iteration, ','.join(map(str, output)) + '\n'))
    if (last_output == output).all():
      done = True
      pattern_matched = True
    else:
      for past_output in past_outputs:
        if (past_outputs == output).all():
          process_file.write('Se ha detectado un ciclo en la iteración {0}\n'.format(iteration))
          done = True  
    last_output = output
    past_outputs.append(last_output)
  process_file.close()
  return (pattern_matched, last_output, iteration)
     
if __name__ == '__main__':

  # Handling command line arguments
  
  parse_weighted_matrix_from_file = False
  
  if len(sys.argv) == 2:
    parse_weighted_matrix_from_file = sys.argv[1] == 'load_weighted_matrix'
  
  NEURON_COUNT = 100
  IMAGE_SIZE = (10, 10)
  
  weighted_matrix = None

  patterns = load_patterns(IMAGE_SIZE)
  
  WEIGHTED_MATRIX_FILE_PATH = 'patterns/weighted_matrix.csv'
    
  if parse_weighted_matrix_from_file:
    print('Cargando matrix de pesos desde archivo "{0}"'.format(WEIGHTED_MATRIX_FILE_PATH))
    weighted_matrix = parse_weighted_matrix(WEIGHTED_MATRIX_FILE_PATH)
    if weighted_matrix is None:
      weighted_matrix = compute_weighted_matrix(patterns, NEURON_COUNT * 1.0)
      if patterns is None or len(patterns) == 0:
        print('Error de usuario')
        exit(1)
    else:
      print('Matrix de pesos cargada de registro.')

  if weighted_matrix is None:
    weighted_matrix = compute_weighted_matrix(patterns, NEURON_COUNT * 1.0)
    if patterns is None or len(patterns) == 0:
        print('Error de usuario')
        exit(1)
    file = open(WEIGHTED_MATRIX_FILE_PATH, 'w')
    for row in weighted_matrix:
      file.write(','.join(map(str, row)) + '\n')
    file.close()
  
  print('Número de patrones:', len(patterns))
  
  pattern = load_pattern('patterns/letra_C_malformada.png', IMAGE_SIZE)
  #print('Patron a asociar:', pattern)
  
  print('Tratando de asociar patron 1 ...')
  (pattern_matched, reconstructed_pattern, iterations) = associate(weighted_matrix, pattern)
  print('Nùmero de iteraciones realizadas:', iterations)
  if pattern_matched:
    generated_image = reconstructed_pattern.reshape(IMAGE_SIZE)
    # TODO Evaluate if the following two lines are convinient
    generated_image[generated_image <=  0] = 0
    generated_image[generated_image ==  1] = 255
    # print(generated_image)
    print('Patron asociado')
    cv2.imshow('Patron recontruido', cv2.resize(generated_image, (IMAGE_SIZE[1] * 25, IMAGE_SIZE[0] * 25), interpolation = cv2.INTER_NEAREST))
    cv2.waitKey(0)
  else:
    print('Patron no asociado')
