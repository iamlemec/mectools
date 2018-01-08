# print iterator progress
def progress(it, per=100, fmt='%s'):
    print(fmt % 'starting')
    i = 0
    for x in it:
        yield x
        i += 1
        if i % per == 0:
            print(fmt % str(i))
    print(fmt % 'done')

