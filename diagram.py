# svg diagram maker

import copy
import numpy as np

# defaults
size_base = (100, 100)
rect_base = (0, 0, 1, 1)

def demangle(k):
    return k.replace('_', '-')

def dict_repr(d):
    return ' '.join([f'{demangle(k)}="{v}"' for k, v in d.items()])

def map_coords(rect0, rect1):
    x0, y0, w0, h0 = rect0
    x1, y1, w1, h1 = rect1
    return (x0 + x1*w0, y0 + y1*h0, w1*w0, h1*h0)

def merge(d1, **d2):
    return {**d1, **d2}

class Context:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def coords(self, rect):
        rect1 = map_coords(self.rect, rect)
        ctx = self.copy()
        ctx.rect = rect1
        return ctx

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

class Element:
    def __init__(self, tag, **attr):
        self.tag = tag
        self.attr = attr

    def __repr__(self):
        attr = dict_repr(self.attr)
        return f'{self.tag}: {attr}'

    def props(self, ctx):
        return self.attr

    def inner(self, ctx):
        return ''

    def svg(self, ctx=None):
        if ctx is None:
            ctx = Context(rect=size_base)

        props = dict_repr(self.props(ctx))
        inner = self.inner(ctx)

        pre = ' ' if len(props) > 0 else ''
        pad = '\n' if len(inner) > 0 else ''

        return f'<{self.tag}{pre}{props}>{pad}{inner}{pad}</{self.tag}>'

class Container(Element):
    def __init__(self, children=None, tag='g', **attr):
        super().__init__(tag=tag, **attr)
        if children is None:
            self.children = []
        else:
            self.children = children

    def add(self, child, rect=rect_base):
        self.children.append((child, rect))
        return self

    def inner(self, ctx):
        return '\n'.join([c.svg(ctx.coords(r)) for c, r in self.children])

class Diagram(Container):
    def __init__(self, children=None, size=size_base, **attr):
        super().__init__(children=children, tag='svg', **attr)
        self.size = size

    def props(self, ctx):
        w, h = self.size
        return merge(self.attr, width=w, height=h)

    def svg(self):
        rect0 = (0, 0) + self.size
        ctx = Context(rect=rect0)
        return Element.svg(self, ctx)

    def save(self, path):
        s = self.svg()
        with open(path, 'w+') as fid:
            fid.write(s)

class Line(Element):
    def __init__(self, *args, **kwargs):
        super().__init__(tag='line', *args, **kwargs)

    def props(self, ctx):
        x1, y1, w1, h1 = ctx.rect
        return merge(self.attr,
            x1=x1,
            y1=y1,
            x2=x1+w1,
            y2=y1+h1,
            stroke_width=1,
            stroke='black',
        )

class SymbolicLine(Element):
    def __init__(self, formula, xlim, ylim=None, N=100, *args, **kwargs):
        super().__init__(tag='path', *args, **kwargs)

        self.xvals = np.linspace(*xlim, N)
        if type(formula) is str:
            self.yvals = eval(formula, {'x': self.xvals})
        else:
            self.yvals = formula(self.xvals)

        self.formula = formula
        self.xlim = xlim
        if ylim is None:
            self.ylim = np.min(self.yvals), np.max(self.yvals)
        else:
            self.ylim = ylim

    def props(self, ctx):
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        xnorm = (self.xvals-xmin)/(xmax-xmin)
        ynorm = (self.yvals-ymin)/(ymax-ymin)

        cx, cy, cw, ch = ctx.rect
        xcoord = cx + xnorm*cw
        ycoord = cy + ynorm*ch

        path = (
            f'M {xcoord[0]},{ycoord[0]}'
            + ' '.join([f'L {x},{y}' for x, y in zip(xcoord[1:], ycoord[1:])])
        )

        return merge(self.attr,
            d=path,
            stroke='black',
        )
