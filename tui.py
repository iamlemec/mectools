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

# evenly space strings
def spaced_strings(texts, width):
    # get string centers
    num = len(texts)
    lens = torch.tensor([len(s) for s in texts])

    # get location span
    loc1 = (torch.linspace(0, width, num) - lens / 2).round().long()
    loc2 = loc1 + lens

    # handle overhang
    over = (-loc1).clamp(min=0) - (loc2 - width).clamp(min=0)
    loc1 += over
    loc2 += over

    # construct strings
    line = [' ' for _ in range(width)]
    for s, i1, i2 in zip(texts, loc1.tolist(), loc2.tolist()):
        for k, i in enumerate(range(i1, i2)):
            line[i] = s[k]

    return ''.join(line)

def label_format(x):
    if x % 1.0 == 0:
        return f'{round(x):d}'
    else:
        return f'{x:.2f}'

def hist(data, bins=10, vmin=None, vmax=None, drop=False, **kwargs):
    # convert to torch
    data = torch.as_tensor(data, dtype=torch.float32)

    # get bounds
    vmin = data.min().item() if vmin is None else vmin
    vmax = data.max().item() if vmax is None else vmax

    # construct histogram
    hist = torch.histc(data, bins=bins, min=vmin, max=vmax)

    # add under/overflow
    if not drop:
        hist[ 0] += (data < vmin).sum().item()
        hist[-1] += (data > vmax).sum().item()

    # construct histogram
    plot = bar(hist, **kwargs)

    # add x-axis labels
    nlabs = max(2, bins // 15)
    xlocs = torch.linspace(vmin, vmax, nlabs)
    labels = [label_format(x) for x in xlocs.tolist()]
    axis = spaced_strings(labels, bins)

    return plot + '\n' + axis
