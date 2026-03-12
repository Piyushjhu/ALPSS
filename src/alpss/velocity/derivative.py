import numpy as np

# function to take the numerical derivative of input array phas (central difference with a 9-point stencil).
# phas is padded so that after smoothing the final velocity trace matches the length of the domain of interest.
# this avoids issues with handling the boundaries in the derivative and later in smoothing.
def num_derivative(phas, window, time_start_idx, time_end_idx, fs):

    # calculate how much padding is needed. half_space padding comes from the length of the smoothing window. the 4
    # is added to account for the 9-point stencil
    half_space = int(np.floor(window / 2))
    pad = half_space + 4

    # Clamp slice indices to array bounds; fill any missing samples with edge
    # values so the array length stays correct for downstream smoothing.
    raw_start = time_start_idx - pad
    raw_end = time_end_idx + pad
    actual_start = max(0, raw_start)
    actual_end = min(len(phas), raw_end)
    left_missing = actual_start - raw_start    # > 0 only when clamped at start
    right_missing = raw_end - actual_end       # > 0 only when clamped at end
    phas_pad = np.pad(
        phas[actual_start:actual_end],
        (left_missing, right_missing),
        mode="edge",
    )

    # calculate the derivative with the 9-point central difference method
    dpdt_pad = np.zeros(phas_pad.shape)
    for i in range(4, len(dpdt_pad) - 4):
        dpdt_pad[i] = ((1 / 280) * phas_pad[i - 4]
                       + (-4 / 105) * phas_pad[i - 3]
                       + (1 / 5) * phas_pad[i - 2]
                       + (-4 / 5) * phas_pad[i - 1]
                       + (4 / 5) * phas_pad[i + 1]
                       + (-1 / 5) * phas_pad[i + 2]
                       + (4 / 105) * phas_pad[i + 3]
                       + (-1 / 280) * phas_pad[i + 4]) \
                      * (fs / (2 * np.pi))

    # output both the padded and un-padded derivatives
    dpdt = dpdt_pad[pad:-pad]
    dpdt_pad = dpdt_pad[4:-4]

    return dpdt, dpdt_pad
