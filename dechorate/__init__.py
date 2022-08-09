import numpy as np

constants = {
    'recording_offset'  : {
        'standard'      :  4444 # samples
    ,   '/rir/000001/1' : 12636
    ,   '/rir/000001/5' :  8540
    ,   '/rir/000010/4' :  8540    
    ,   '/rir/001000/6' :  8540
    ,   '/rir/011100/6' : 12636
    },
    'Fs' : 48000, # samples / seconds
    'rir_length': 19556, # samples
    'room_size': [5.705, 5.965, 2.355],  # meters
    'offset_beacon' : [0.08, 0.095, 0], # meters [12-5+1, 13.5-5+1] = offset panels ceiling + offset panes wall + beacon
    'datasets' : ['000000', '010000', '011000', '011100', '011110', '011111', '001000', '000100', '000010', '000001', '020002'],
    'wall_dataset': ['f', 'c', 'w', 's', 'e', 'n'],
    'refl_order_pyroom': ['d', 'w', 'e', 's', 'n', 'f', 'c'],
    'refl_order_calibr': ['d', 'c', 'f', 'w', 's', 'e', 'n'],
    'room_temperature' : 24,     # temperature
    'room_humidity' : 80,
    'speed_of_sound' : 346.98,  # speed of sound
    'silence_ids': [99],
    'nse_ids': np.arange(4),
    'src_ids': np.arange(9),
    'mic_ids': np.arange(31),
    'signals' : ['chirp', 'silence', 'babble', 'speech', 'noise'],
    'rir_processing': {
        'Fs': 48000,
        'stimulus': 'exp_sine_sweep',
        'n_seconds': 10,
        'amplitude': 0.7,
        'n_repetitions': 3,
        'silence_at_start': 2,
        'silence_at_end': 2,
        'sweeprange': [100, 14e3],
    }
}
