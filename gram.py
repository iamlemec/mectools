# diagram creator

import math

# native format is SVG + CSS
# use cairosvg to convert to PDF/PNG

def gen_css(sel, rules):
    css = f'{sel} {{\n'
    for k, v in rules.items():
        css += f'{k}: {v};'
    css += f'}}'
    return css

def gen_attr(attr):
    props = {k.replace('_', '-'): v for k, v in attr.items()}
    return ' '.join([f'{k}="{v}"' for k, v in props.items()])

class Canvas:
    def __init__(self, root=None, box=(0, 0, 100, 100), **attr):
        self.box = box
        self.root = root
        self.attr = attr

    def set_root(self, root):
        self.root = root

    def render(self, fmt='svg'):
        core = self.root.render()
        attr0 = {
            'viewBox': ' '.join([f'{x}' for x in self.box])
        }
        attr1 = dict(attr0, **self.attr)
        props = gen_attr(attr1)
        svg = f'<svg {props}>\n{core}\n</svg>'
        if fmt == 'svg':
            return svg
        else:
            raise

    def save(self, filename, fmt='svg'):
        svg = self.render(fmt='svg')
        if fmt == 'svg':
            with open(filename, 'w+') as fout:
                fout.write(svg)
        else:
            raise

# shell element class
class Element:
    def __init__(self, tag, **attr):
        self.tag = tag
        self.attr = attr

    def props(self):
        return {}

    def render(self, inner=None):
        attr0 = self.props()
        attr1 = dict(attr0, **self.attr)
        props = gen_attr(attr1)
        if inner is None:
            return f'<{self.tag} {props} />'
        else:
            return f'<{self.tag} {props}>\n{inner}\n</{self.tag}>'

class Area(Element):
    def __init__(self, children=[], **attr):
        super().__init__()
        self.children = children

    def add_child(self, child):
        self.children.append(child)

    def render(self):
        pass

class Rows(Element):
    def __init__(self, rows=[], **attr):
        super().__init__()
        self.rows = rows

    def add_row(self, row):
        self.rows.append(row)

    def render(self):
        pass

class Cols(Element):
    def __init__(self, cols=[], **kwargs):
        super().__init__()
        self.cols = cols

    def add_row(self, col):
        self.cols.append(col)

    def render(self):
        pass

class Grid(Element):
    pass

class Text(Element):
    def __init__(self, x, y, text, **attr):
        super().__init__('text', **attr)
        self.x = x
        self.y = y
        self.text = text

    def props(self):
        return {
            'x': self.x,
            'y': self.y
        }

    def render(self):
        return super().render(self.text)

class Line(Element):
    def __init__(self, x1, y1, x2, y2, **attr):
        super().__init__('line', **attr)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def props(self):
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'stroke': 'black',
            'stroke-width': 0.5
        }

class Path(Element):
    def __init__(self, points, **attr):
        super().__init__('path', **attr)
        self.points = points

    def props(self):
        x0, y0 = self.points[0]
        data = [f'M {x0},{y0}'] + [f'L {x},{y}' for x, y in self.points[1:]]
        return {
            'd': ' '.join(data),
            'stroke': 'black',
            'stroke-width': 0.5,
            'fill': 'none'
        }

class Arrow(Element):
    def __init__(self, x1, y1, x2, y2, theta=45, length=1, line_attr={}, arrow_attr={}, **attr):
        super().__init__('g', **attr)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.theta = theta
        self.length = length
        self.line_attr = line_attr
        self.arrow_attr= arrow_attr

        self.generate()

    def generate(self):
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
        rtheta = math.radians(self.theta)

        gamma = math.atan((x2-x1)/(y2-y1)) # angle of line
        etah = gamma - rtheta # residual angle high
        etal = math.radians(180) - gamma - rtheta # residual angle low
        dxh, dyh = self.length*math.cos(etah), self.length*math.sin(etah)
        dxl, dyl = self.length*math.cos(etal), self.length*math.sin(etal)

        # print(f'gamma = {math.degrees(gamma)}')
        # print(f'etah = {math.degrees(etah)}, etal = {math.degrees(etal)}')
        # print(f'dxh = {dxh}, dyh = {dyh}')
        # print(f'dxl = {dxl}, dyl = {dyl}')

        self.line = Line(x1, y1, x2, y2, **self.line_attr)
        self.arrow = Path([(x2-dxh, y2-dyh), (x2, y2), (x2-dxl, y2-dyl)], **self.arrow_attr)

    def props(self):
        return {}

    def render(self):
        inner = self.line.render() + '\n' + self.arrow.render()
        return super().render(inner)

class Image(Element):
    pass

class Plot(Area):
    pass