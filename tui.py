# terminal plotting

from math import ceil, isnan, isinf
import torch
import rich
import rich.panel

# unicode block characters (one eighth increments)
empty, full = 0x00020, 0x02588
upper_blocks = [0x00020, 0x02594, 0x1FB82, 0x1FB83, 0x02580, 0x1FB84, 0x1FB85, 0x1FB86, 0x02588]
lower_blocks = [0x00020, 0x02581, 0x02582, 0x02583, 0x02584, 0x02585, 0x02586, 0x02587, 0x02588]

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def block_code(x, zero, height):
    # check for null or inf
    if isnan(height):
        return empty
    elif isinf(height):
        if height > zero:
            return full if x >= zero else empty
        else:
            return full if x < zero else empty

    # get block index
    if height >= zero:
        index8 = round(8 * clamp(height - x, 0, 1)) if x >= zero else 0
        blocks = lower_blocks
    else:
        # we use (x+1) because that's the top of the block
        index8 = round(8 * clamp((x+1) - height, 0, 1)) if x < zero else 0
        blocks = upper_blocks

    # return block
    return blocks[index8]

def block_char(x, zero, height):
    return chr(block_code(x, zero, height))

def nanmax(x):
    return torch.max(x[x.isfinite()]).item()

def nanmin(x):
    return torch.min(x[x.isfinite()]).item()

def bar(data, ymin=None, ymax=None, height=20, zero=None):
    # convert to torch
    data = torch.as_tensor(data)
    width = data.numel()

    # get bounds
    ymin = nanmin(data) if ymin is None else ymin
    ymax = nanmax(data) if ymax is None else ymax
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
    ax = abs(x)
    isint = x % 1.0 == 0
    if isint or ax > 10:
        digits = 0
    elif ax > 1:
        digits = 1
    else: # ax < 1
        digits = 2
    if isint:
        return f'{round(x):d}'
    else:
        return f'{round(x, digits):.{digits}f}'

def hist(
    data, bins=10, vmin=None, vmax=None, drop=False, labels=None,
    fancy=True, log=False, title='histogram', **kwargs
):
    # convert to torch
    data = torch.as_tensor(data, dtype=torch.float32)

    # get bounds
    vmin = nanmin(data) if vmin is None else vmin
    vmax = nanmax(data) if vmax is None else vmax

    # construct histogram
    hist = torch.histc(data, bins=bins, min=vmin, max=vmax)

    # add under/overflow
    if not drop:
        hist[ 0] += (data < vmin).sum().item()
        hist[-1] += (data > vmax).sum().item()

    # apply log scale
    if log:
        hist = torch.where(hist > 0, hist.log10(), torch.nan)

    # construct histogram
    plot = bar(hist, **kwargs)

    # add x-axis labels
    if labels is not False:
        nlabs = max(2, bins // 15) if labels is None else labels
        xlocs = torch.linspace(vmin, vmax, nlabs)
        labels = [label_format(x) for x in xlocs.tolist()]
        axis = spaced_strings(labels, bins)
        plot = plot + '\n' + axis

    # use rich if fancy
    if fancy:
        return rich.panel.Panel(plot, expand=False, title=title)
    else:
        return plot
