import numpy as np
import pyroomacoustics as pra
import soundfile as sf

from src.utils.dsp_utils import normalize, center


def callback_mix(premix, snr=0, sir=0, ref_mic=0, n_src=None, n_tgt=None):

    I, J, N = premix.shape

    # first normalize all separate recording to have unit power at microphone one
    p_mic_ref = np.std(premix[:, ref_mic, :], axis=1)
    premix /= p_mic_ref[:, None, None]

    # now compute the power of interference signal needed to achieve desired SIR
    sigma_i = np.sqrt(10 ** (- sir / 10) / (n_src - n_tgt))
    premix[n_tgt:n_src, :, :] *= sigma_i

    # compute noise variance
    noise = np.random.randn(I, N)
    if snr < 100:
        sigma_n = np.sqrt(10 ** (- snr / 10))
        noise = noise / np.std(noise[:, ref_mic])
    else:
        sigma_n = 0

    # Mix down the recorded signals
    mix = np.sum(premix[:n_src, :, :], axis=0) + sigma_n * noise
    premix = premix
    return mix


def audio_scene_to_pra_room(audio_scene):

    Fs = audio_scene['Fs']
    Korder = audio_scene['echo_order']

    src1_pos = audio_scene['src_pos']
    src2_pos = audio_scene['src2_pos']
    mic1_pos = audio_scene['mic1_pos']
    mic2_pos = audio_scene['mic2_pos']
    absorption = {
        'east': audio_scene['abs_wall_east'],
        'west': audio_scene['abs_wall_west'],
        'north': audio_scene['abs_wall_north'],
        'south': audio_scene['abs_wall_south'],
        'ceiling': audio_scene['abs_wall_ceiling'],
        'floor': audio_scene['abs_wall_floor'],
    }

    # Sources position and room design
    src1_pos = list(src1_pos)
    src2_pos = list(src2_pos)
    room_size = list(audio_scene['RoomSize'])

    # Create PRA auditory scene
    room = pra.ShoeBox(room_size,
                       fs=Fs, max_order=Korder,
                       absorption=absorption)

    s1, Fs = sf.read(audio_scene['path_to_wav1'])
    s2, Fs = sf.read(audio_scene['path_to_wav2'])

    s1 = normalize(center(s1))
    s2 = normalize(center(s2))

    room.add_source(src1_pos, signal=s1, delay=3)
    room.add_source(src2_pos, signal=s2, delay=1)

    # Microphone array design parameters
    room.add_microphone_array(
        pra.MicrophoneArray(
            np.concatenate([mic1_pos[:, None], mic2_pos[:, None]], axis=1), Fs))
    return room


def from_audio_scene_to_sound_images(audio_scene):
    # Run Image Method simulation
    room.image_source_model()
    callback_mix_kwargs = {
        'snr': audio_scene['snr'],
        'sir': audio_scene['sir'],
        'n_src': 2,
        'n_tgt': 1,
        'ref_mic': 0,
    }
    premix = room.simulate(
        callback_mix=callback_mix,
        callback_mix_kwargs=callback_mix_kwargs,
        return_premix=True,
    )   # J x I x N
    premix = premix.transpose([2, 1, 0])
    mix = room.mic_array.signals.T

    N = min(premix.shape[0], mix.shape[0])
    premix = premix[:N, :, :]
    mix = mix[:N, :]

    return mix, premix


def compute_partial_rir(mic, position, damping, visibility, Fs, t0=0., t_max=None):
    '''
    Compute the room impulse response between the source
    and the microphone whose position is given as an
    argument.
    '''

    # fractional delay length
    fdl = pra.constants.get('frac_delay_length')
    fdl2 = (fdl-1) // 2

    # compute the distance
    dist = np.linalg.norm(mic - position, axis=0)[None]
    time = dist / pra.constants.get('c') + t0
    alpha = damping / (4.*np.pi*dist)

    # the number of samples needed
    if t_max is None:
        # we give a little bit of time to the sinc to decay anyway
        N = np.ceil((1.05*time.max() - t0) * Fs)
    else:
        N = np.ceil((t_max - t0) * Fs)

    N += fdl

    t = np.arange(N) / float(Fs)
    ir = np.zeros(t.shape)

    from pyroomacoustics.utilities import fractional_delay

    for i in range(time.shape[0]):
        if visibility[i] == 1:
            time_ip = int(np.round(Fs * time[i]))
            time_fp = (Fs * time[i]) - time_ip
            ir[time_ip-fdl2:time_ip+fdl2+1] += alpha[i] * \
                fractional_delay(time_fp)
    return ir


def get_partial_atf_and_rir(audio_scene, F):
    mic1_pos = audio_scene['mic1_pos']
    mic2_pos = audio_scene['mic2_pos']
    src_pos = audio_scene['src_pos']
    Fs = audio_scene['Fs']
    max_order = np.min([np.cbrt(K), 1])
    room = pra.ShoeBox(audio_scene['room_size'],
                       fs=audio_scene['Fs'], max_order=max_order)
    room.add_microphone_array(
        pra.MicrophoneArray(
            np.concatenate([mic1_pos[:, None], mic2_pos[:, None]], axis=1), Fs))
    room.add_source(src_pos)

    I = room.mic_array.R.shape[-1]

    freqs = np.linspace(0, Fs//2, F)
    omegas = 2*np.pi*freqs

    room.image_source_model(use_libroom=False)

    a_im = np.zeros([L, I, J, Kim])
    A_im = np.zeros([F, I, J, Kim], dtype=np.complex64)
    for i in range(I):
        for j in range(J):
            images = room_dok.sources[j].images
            center = room_dok.mic_array.center
            distances = np.linalg.norm(images - center, axis=0)
            # order in loc
            ordering = np.argsort(distances)[:Kim]
            for o, k in enumerate(ordering):
                tmp = compute_partial_rir(room_dok.mic_array.R[:, i],
                                          room_dok.sources[j].images[:, k],
                                          room_dok.sources[j].damping[k],
                                          room_dok.visibility[j][i],
                                          room_dok.fs, room_dok.t0)
                l = min(Lh, len(tmp))
                a_im[:l, i, j, o] = tmp[:l]
                A_im[:, i, j, o] = np.fft.rfft(tmp[:l], n=2*(F-1))

    for i in range(I):
        images = room.sources[0].images
        center = room.mic_array.center
        distances = np.linalg.norm(
            images - room.mic_array.R[:, i, None], axis=0)
        ordering = np.argsort(distances)[:K]
        for o, k in enumerate(ordering):
            amp = room.sources[0].damping[k] / (4 * np.pi * distances[k])
            tau = distances[k]/343
            A_im[:, i, o] = amp * np.exp(-1j*omegas*tau)

    return A_im
