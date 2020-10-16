constants = {
    'recording_offset': 4444,  # samples
    'Fs' : 48000, # samples / seconds
    'rir_length': 19556, # samples
    'room_size': [5.705, 5.965, 2.355],  # meters
    'offset_beacon' : [0.08, 0.095, 0], # meters [12-5+1, 13.5-5+1] = offset panels ceiling + offset panes wall + beacon
    # 'room_size': [556.5, 5.680, ]
    'datasets' : ['000000', '010000', '011000', '011100', '011110', '011111', '001000', '000100', '000010', '000001', '0F000F'],
    'wall_dataset': ['f', 'c', 'w', 's', 'e', 'n'],
    'refl_order_pyroom': ['d', 'w', 'e', 's', 'n', 'f', 'c'],
    'refl_order_calibr': ['d', 'c', 'f', 'w', 's', 'e', 'n'],
    'room_temperature' : 24,     # temperature
    'speed_of_sound' : 331.3 + 0.606 * 24,  # speed of sound
}
