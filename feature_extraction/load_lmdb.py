import os
import lmdb
import numpy as np
from tqdm import tqdm

def write_to_lmdb(directory, lmdb_path):
    map_size = 1099511627776  # 1 TB (adjust as needed)
    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for filename in tqdm(os.listdir(directory)):
            if filename.endswith('.npy'):
                key = os.path.splitext(filename)[0].encode('ascii')
                file_path = os.path.join(directory, filename)
                with open(file_path, 'rb') as f:
                    value = np.load(f)
                    value_bytes = value.tobytes()
                    txn.put(key, value_bytes)

    env.close()

# if __name__ == "__main__":
#     directory_path = "dataset/features_mosei/hubert-large-ls960-ft-FRA_-3"
#     lmdb_file_path = "dataset/features_mosei/hubert-large-ls960-ft-FRA_-3.lmdb"

#     write_to_lmdb(directory_path, lmdb_file_path)


def lmdb_to_npy(lmdb_path, output_directory, dtype=np.float32, shape=None):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    env = lmdb.open(lmdb_path, readonly=True)

    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in tqdm(cursor):
            array = np.frombuffer(value, dtype=dtype)
            if shape is not None:
                array = array.reshape(shape)
            filename = key.decode('ascii') + '.npy'
            output_path = os.path.join(output_directory, filename)
            np.save(output_path, array)

    env.close()

if __name__ == "__main__":
    lmdb_file_path = "dataset/features_mosei/wavlm-large-FRA_-5.lmdb"
    output_directory_path = "dataset/features_mosei/wavlm-large-FRA_-5"

    # Specify dtype and shape if known
    lmdb_to_npy(lmdb_file_path, output_directory_path, dtype=np.float32, shape=(-1,1024))