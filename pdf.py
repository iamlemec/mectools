import re
import fire
import subprocess

def strip_cruft(text):
    text = re.sub(r'(?:^|\n)[^a-zA-Z]+(?:$|\n)', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def pdf_to_text(src, strip=False):
    res = subprocess.run(['pdftotext', src, '-'], stdout=subprocess.PIPE)
    if res.returncode == 0:
        text = res.stdout.decode()
        if strip:
            text = strip_cruft(text)
        return text

class Main:
    def __init__(self, src, dst=None):
        self.src = src
        self.dst = dst

    def text(self, strip=False):
        text = pdf_to_text(self.src, strip=strip)
        if text is None:
            print('Conversion failed')
            return

        if self.dst is None:
            print(text)
        else:
            with open(self.dst, 'w+') as fid:
                fid.write(text)

    def length(self, strip=True):
        text = pdf_to_text(self.src, strip=strip)
        if text is None:
            print('Conversion failed')
            return

        chars = len(text)
        words = len(text.split())
        print(f'chars: {chars}, words: {words}')

if __name__ == '__main__':
    fire.Fire(Main)
