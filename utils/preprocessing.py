from .helpers import subsample_list


def _blosc_opts(complevel=1, complib='blosc:zstd', shuffle='byte'):
    shuffle = 2 if shuffle == 'bit' else 1 if shuffle == 'byte' else 0
    compressors = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
    complib = ['blosc:' + c for c in compressors].index(complib)
    args = {
        'compression': 32001,
        'compression_opts': (0, 0, 0, 0, complevel, shuffle, complib),
    }
    if shuffle > 0:
        # Do not use h5py shuffle if blosc shuffle is enabled.
        args['shuffle'] = False
    return args


def subsample_sequence(split_path, ratio):
    """Subsample the sequence under a folder by a given ratio."""
    seq_dirs = sorted([p for p in split_path.iterdir()])
    print(f'Found {len(seq_dirs)} sequences in {str(split_path)}')
    # may need to sub-sample training seqs
    if 0. < ratio < 1.:
        num = round(len(seq_dirs) * ratio)
        seq_dirs = subsample_list(seq_dirs, num)
        assert len(seq_dirs) == num
        print(f'Using {ratio*100}% of data --> {len(seq_dirs)} sequences')
    return seq_dirs
