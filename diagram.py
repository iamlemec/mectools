# svg diagram maker

import copy

size_base = (100, 100)
rect_base = (0, 0, 1, 1)

def demangle(k):
    return k.replace('_', '-')

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

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

class Element:
    def __init__(self, tag, rect=rect_base, **attr):
        self.tag = tag
        self.rect = rect
        self.attr = attr

    def coords(self, ctx):
        return map_coords(ctx.rect, self.rect)

    def props(self, ctx):
        return self.attr

    def inner(self, ctx):
        return ''

    def svg(self, ctx):
        if ctx is None:
            ctx = Context()

        props = {demangle(k): v for k, v in self.props(ctx).items()}
        inner = self.inner(ctx)

        if len(props) > 0:
            props = ' ' + ' '.join([f'{k}="{v}"' for k, v in props.items()])
        else:
            props = ''

        if len(inner) > 0:
            inner = f'\n{inner}\n'

        return f'<{self.tag}{props}>{inner}</{self.tag}>'

class Container(Element):
    def __init__(self, children=[], tag='g', rect=rect_base, **attr):
        super().__init__(tag=tag, rect=rect, **attr)
        self.children = children

    def add(self, child):
        self.children.append(child)

    def inner(self, ctx):
        ctx1 = ctx.copy()
        ctx1.rect = map_coords(ctx.rect, self.rect)
        return '\n'.join([c.svg(ctx1) for c in self.children])

class Diagram(Container):
    def __init__(self, children=[], size=size_base, rect=rect_base, **attr):
        super().__init__(children, tag='svg', rect=rect, **attr)
        self.size = size

    def props(self, ctx):
        w, h = self.size
        return merge(self.attr, width=w, height=h)

    def svg(self):
        rect0 = (0, 0) + self.size
        ctx = Context(rect=rect0)
        return super().svg(ctx)

class Line(Element):
    def __init__(self, *args, **kwargs):
        super().__init__(tag='line', *args, **kwargs)

    def props(self, ctx):
        x1, y1, w1, h1 = self.coords(ctx)
        return merge(self.attr,
            x1=x1,
            y1=y1,
            x2=x1+w1,
            y2=y1+h1,
            stroke_width=1,
            stroke='black',
        )

class SymbolicLine(Element):
    def __init__(self, formula, *args, **kwargs):
        super().__init__(tag='path', *args, **kwargs)
        self.formula = formula
