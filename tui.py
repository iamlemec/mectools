# terminal plotting

from math import ceil
import torch

# unicode block characters (one eighth increments)
upper_blocks = [0x00020, 0x02594, 0x1FB82, 0x1FB83, 0x02580, 0x1FB84, 0x1FB85, 0x1FB86, 0x02588]
lower_blocks = [0x00020, 0x02581, 0x02582, 0x02583, 0x02584, 0x02585, 0x02586, 0x02587, 0x02588]

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def block_code(x, zero, height):
    if height >= zero:
        index8 = round(8 * clamp(height - x, 0, 1)) if x >= zero else 0
        blocks = lower_blocks
    else:
        # we use (x+1) because that's the top of the block
        index8 = round(8 * clamp((x+1) - height, 0, 1)) if x < zero else 0
        blocks = upper_blocks
    return blocks[index8]

def block_char(x, zero, height):
    return chr(block_code(x, zero, height))

def bar(data, ymin=None, ymax=None, height=20, zero=None):
    # convert to torch
    data = torch.as_tensor(data)
    width = data.numel()

    # get bounds
    ymin = data.min().item() if ymin is None else ymin
    ymax = data.max().item() if ymax is None else ymax
    zero = ymin if zero is None else zero

    # scale data
    zero_height = height * (zero - ymin) / (ymax - ymin)
    data_height = height * (data - ymin) / (ymax - ymin)

    # adjust so that zero is on a block boundary
    zero_offset = ceil(zero_height) - zero_height
    if zero_offset > 0:
        zero_height += zero_offset
        data_height += zero_offset
        height += 1

    # construct bar chars
    cols = [
        [
            block_char(x, zero_height, dh) for x in reversed(range(height))
        ] for dh in data_height.tolist()
    ]

    # transpose and join
    return '\n'.join(''.join(row) for row in zip(*cols))
