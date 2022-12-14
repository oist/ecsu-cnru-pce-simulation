"""
Utility functions for vector manipulation etc.
"""

import json
import string
import numpy as np
from numpy import pi
import os
import hashlib
import codecs
import pickle

# CONSTANTS
TWO_PI = 2 * pi
RANDOM_CHAR_SET = string.ascii_uppercase + string.digits


def linmap(vin, rin, rout):
    """
    Map a vector between 2 ranges.
    :param vin: input vector to be mapped
    :param rin: range of vin to map from
    :param rout: range to map to
    :return: mapped output vector
    :rtype np.ndarray
    """
    a = rin[0]
    b = rin[1]
    c = rout[0]
    d = rout[1]
    return ((c + d) + (d - c) * ((2 * vin - (a + b)) / (b - a))) / 2

def discretize(a, bins, min_v=0, max_v=1):
    a[a > max_v] = max_v
    a[a < min_v] = min_v
    bins = np.linspace(min_v, max_v, bins)
    return np.digitize(a, bins, right=True)


def random_string(size=5):
    '''
    generates random alphanumeric string
    '''
    return ''.join(np.random.choice(RANDOM_CHAR_SET) for _ in range(size))


def modulo_radians(theta):
    '''
    ensure that the angle is between 0 and 2*pi
    0 <= result_theta < 2*pi
    '''
    return theta % TWO_PI


def angle_in_range(theta, low, high):
    '''
    assume that all angles alpha in args:
    0 <= alpha < 2*pi
    '''
    if low < theta < high:
        return True
    if theta > low:
        return high < low  # wrapping up of high above zero
    if theta < high:
        return low > high  # wrapping up of low below zero
    return False


def rotate_cw_matrix(theta):
    '''
    Returns a rotation clock-wise matrix
    e.g.,
    v = np([np.sqrt(3),1])  # vector of length 2 at 30 (pi/6) angle
    theta = pi/6
    r = rotate_cw_matrix(theta)
    # --> np.dot(v,r) = [2,0]
    '''
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def add_noise(vector, random_state, noise_level):
    return vector + noise_level * random_state.normal(0, 1, vector.shape)


def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    length = np.linalg.norm(vector)
    if length == 0:
        return np.zeros(2)
    return vector / length

def angle_between(v1, v2):
    # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def make_rand_vector(dims, random_state):
    """
    Generate a random unit vector.  This works by first generating a vector each of whose elements
    is a random Gaussian and then normalizing the resulting vector.
    """
    vec = random_state.normal(0, 1, dims)
    mag = sum(vec ** 2) ** .5
    return vec / mag


def save_json_numpy_data(data, file_path):
    import json
    from pyevolver.json_numpy import NumpyListJsonEncoder
    json.dump(
        data,
        open(file_path, 'w'),
        indent=3,
        cls=NumpyListJsonEncoder
    )


def random_int(random_state=None, size=None):
    if random_state is None:
        return np.random.randint(0, 2147483647, size)
    else:
        return random_state.randint(0, 2147483647, size)


def make_dir_if_not_exists_or_replace(dir_path):
    import shutil
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        # assert os.path.isdir(dir_path), 'Path {} is not a directory'.format(dir_path)
        # return
    os.makedirs(dir_path)

def make_dir_if_not_exists(dir_path):
    if os.path.exists(dir_path):        
        return
    else:
        os.makedirs(dir_path)


def assert_string_in_values(s, s_name, values):
    assert s in values, '{} should be one of the following: {}. Given value: {}'.format(s_name, values, s)

def get_numpy_signature(arr):
    hex_hash = hashlib.sha1(arr).hexdigest() 
    sign = codecs.encode(
        codecs.decode(hex_hash, 'hex'), 
        'base64'
    ).decode()[:5]
    return sign    

def genotype_pair_distance(a, b):    
    from pce.params import EVOLVE_GENE_RANGE
    a_norm = linmap(a, EVOLVE_GENE_RANGE, (0,1))
    b_norm = linmap(b, EVOLVE_GENE_RANGE, (0,1))
    diff = a_norm - b_norm
    dist = np.linalg.norm(diff) 
    # same as scipy.spatial.distance.euclidean(a_norm, b_norm)
    max_dist = np.sqrt(len(a))    
    dist_norm = linmap(dist, (0,max_dist), (0,1))
    assert 0 <= dist_norm <= 1
    return dist_norm

def genotype_group_distance(group_genotype):    
    from pce.params import EVOLVE_GENE_RANGE
    from sklearn.metrics.pairwise import pairwise_distances  
    genotype_length = group_genotype.shape[1]
    population_norm = linmap(group_genotype, EVOLVE_GENE_RANGE, (0,1))    
    dist = pairwise_distances(population_norm)
    max_dist = np.sqrt(genotype_length) # length of unary vector of dim = genotype_length
    dist_norm = linmap(dist, (0,max_dist), (0,1))
    assert 0 <= dist_norm.all() <= 1
    return dist_norm


def save_data_to_pickle(data, pickle_file):
    with open(pickle_file, 'wb') as handle:
            pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)

def load_data_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as handle:
        return pickle.load(handle)		
    
def am_i_on_deigo():
    import socket
    return socket.gethostname().startswith('deigo')

def moving_average(a, w=3) :
    ret = np.cumsum(a, dtype=float)
    ret[w:] = ret[w:] - ret[:-w]
    return ret[w - 1:] / w

def is_int(s):
    try:
        int(s)
    except ValueError:
        return False
    return True

def is_float(s):
    try:
        float(s)
    except ValueError:
        return False
    return True

