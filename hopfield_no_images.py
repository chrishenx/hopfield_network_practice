import numpy as np

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
  print('U({0}) = {1}'.format(iteration, inhibition_state))
  print('Y({0}) = {1}\n'.format(iteration, last_output))
  while not done:
    iteration += 1
    inhibition_state = np.dot(weighted_matrix, last_output)
    print('U({0}) = {1}'.format(iteration, inhibition_state))
    output = transfer_function(inhibition_state)
    print('Y({0}) = {1}\n'.format(iteration, output))
    if (last_output == output).all():
      done = True
      pattern_matched = True
    else:
      for past_output in past_outputs:
        if (past_outputs == output).all():
          done = True  
    last_output = output
    past_outputs.append(last_output)
  return (pattern_matched, last_output, iteration)
  
if __name__ == "__main__":
  print('Red Hopfield por Christian Gonzále León\n')

  try:
    neuron_count = int(input('Número de neuronas en la red: '))    
    done = False
    patterns = []
    
    while not done:
      pattern_count = len(patterns)
      print('\nIngresar valores del patrón C{0}'.format(pattern_count))
      pattern = np.zeros(neuron_count, dtype=int)
      neuron = 0
      while neuron < neuron_count:
        try:
          pattern[neuron] = int(input('  C{0}{1} = '.format(pattern_count, neuron)))
          neuron += 1
        except ValueError:
          pass
      patterns.append(pattern)
      done = input('Agregar otro patron? [Si=ENTER, No=Otro]: ') != ''
    
    weighted_matrix = compute_weighted_matrix(patterns, neuron_count)
    print('\nMatriz de pesos:\n\n{0}\n'.format(weighted_matrix))
    
    input('Presiona ENTER para continuar . . .')
    
    done = False
    while not done:
      print('\nIngresar patron de prueba:')
      neuron = 0
      test_pattern = np.zeros(neuron_count, dtype=int)
      while neuron < neuron_count:
        try:
          test_pattern[neuron] = int(input('  X{0} = '.format(neuron)))
          neuron += 1
        except ValueError:
          pass
      print('\nProceso de asociación:\n')
      (pattern_matched, last_output, iteration) = associate(weighted_matrix, test_pattern)
      print()
      if pattern_matched:
        print('Patron asociado en {0} iteraciones'.format(iteration))
      else:
        print('Patron no asociado.')
        print('Ciclo encontrado en iteración {0}'.format(iteration))
      print('Último patron generado:', last_output)
      done = input('\nProbar con otro patron? [Si=ENTER, No=Otro]: ') != ''
    
  except ValueError:
    print('Error en el ingreso de datos')