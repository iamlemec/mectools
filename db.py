# database tools (sqlite3)

import sqlite3
import pandas as pd
import operator as op

def unfurl(ret,idx=[0]):
    if type(idx) != list: idx = [idx]
    return op.itemgetter(*idx)(zip(*ret))

class Connection:
    def __init__(self, db=':memory:'):
        self.db = db
        self.con = sqlite3.connect(db)

    def __del__(self):
        pass

    def close(self):
        self.con.close()

    def commit(self):
        self.con.commit()

    def tables(self, output=True):
        ret = self.execa('select name from sqlite_master where type="table"')
        if output:
            print('\n'.join(sorted([tn for (tn,) in ret])))
        else:
            return ret

    def schema(self, name, output=True):
        ret = self.execa(f'pragma table_info("{name}")')
        if output:
            print('\n'.join([f'{t:<10s} {n:s}' for (i, n, t, _, _, _) in ret]))
        else:
            return ret

    def size(self, name):
        ret = self.execa(f'select count(*) from {name}')

    def exec(self, cmd, *args, commit=False, fetch=None, fetchall=False):
        ret = self.con.execute(cmd, *args)
        if commit:
            self.commit()
        if fetchall:
            ret = ret.fetchall()
        elif fetch is not None:
            ret = ret.fetchmany(fetch)
        return ret

    def execa(self, cmd, *args, **kwargs):
        kwargs['fetchall'] = True
        return self.exec(cmd, *args, **kwargs)

    def execn(self, cmd, n, *args, **kwargs):
        kwargs['fetch'] = n
        return self.exec(cmd, *args, **kwargs)

    def table(self, name, columns=None, cond=None, frame=False, **kwargs):
        if columns is None:
            cols = '*'
        else:
            cols = ','.join(columns)
        cmd = f'select {cols} from {name}'
        if cond is not None:
            cmd += f' {cond}'
        if frame:
            return pd.read_sql(cmd, self.con, **kwargs)
        else:
            return zip(*self.execa(cmd, **kwargs))

def connect(db=None):
    kwargs = {'db': db} if db is not None else {}
    return Connection(**kwargs)

def table_op(tname, schema):
    def wrap(f):
        def f1(**kwargs):
            if 'db' in kwargs:
                con = sqlite3.connect(kwargs.pop('db'))
                close = True
            else:
                con = kwargs.pop('con')
                close = False
            if 'cur' in kwargs:
                cur = kwargs.pop('cur')
            else:
                cur = con.cursor()

            if kwargs.pop('clobber', False):
                cur.execute(f'drop table if exists {tname}')
            cur.execute('create table if not exists %s as %s' % (tname, schema))

            f(cur,**kwargs)

            con.commit()
            if close:
                con.close()
        return f1
    return wrap

class ChunkInserter:
    def __init__(self, con, table=None, cmd=None, cur=None, chunk_size=1000, wild='?', output=False):
        if table is None and cmd is None:
            raise('Must specify either table or cmd')

        self.con = con
        self.cur = cur if cur is not None else con.cursor()
        self.table = table
        self.wild = wild
        self.cmd = cmd
        self.chunk_size = chunk_size
        self.output = output
        self.items = []
        self.i = 0

    def insert(self,*args):
        self.items.append(args)
        if len(self.items) >= self.chunk_size:
            self.commit()
            return True
        else:
            return False

    def insertmany(self,args):
        self.items += args
        if len(self.items) >= self.chunk_size:
            self.commit()
            return True
        else:
            return False

    def commit(self):
        self.i += 1
        if len(self.items) == 0:
            return
        if self.cmd is None:
            nargs = len(self.items[0])
            sign = ','.join(nargs*self.wild)
            self.cmd = 'insert or replace into %s values (%s)' % (self.table, sign)
        if self.output:
            print('Committing chunk %d (%d)' % (self.i,len(self.items)))
        self.cur.executemany(self.cmd,self.items)
        self.con.commit()
        self.items = []

class DummyInserter:
    def __init__(self, *args, chunk_size=1000, output=True, **kwargs):
        self.chunk_size = chunk_size
        self.output = output
        self.last = None
        self.i = 0

    def insert(self,*args):
        self.last = args
        self.i += 1
        if self.i >= self.chunk_size:
            self.commit()
            return True
        else:
            return False

    def insertmany(self,args):
        if len(args) > 0:
            self.last = args[-1]
        self.i += len(args)
        if self.i >= self.chunk_size:
            self.commit()
            return True
        else:
            return False

    def commit(self):
        if self.output:
            print(self.last)
        self.i = 0
