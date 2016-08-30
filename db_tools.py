# database tools (sqlite3)

import sqlite3

def tables(cur):
    ret = cur.execute('select name from sqlite_master where type=\'table\'')
    print('\n'.join([tn for (tn,) in ret.fetchall()]))

def table_info(cur,name):
    ret = cur.execute('pragma table_info(\'%s\')' % name)
    print('\n'.join(['{:<10s} {:s}'.format(t,n) for (i,n,t,_,_,_) in ret.fetchall()]))

def table_op(tname,schema):
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

            if kwargs.pop('clobber',False):
                cur.execute('drop table if exists %s' % tname)
            cur.execute('create table if not exists %s as %s' % (tname,schema))

            f(cur,**kwargs)

            con.commit()
            if close:
                con.close()
        return f1
    return wrap

class ChunkInserter:
    def __init__(self,con,table=None,cmd=None,cur=None,chunk_size=1000,output=False):
        if table is None and cmd is None:
            raise('Must specify either table or cmd')

        self.con = con
        self.cur = cur if cur is not None else con.cursor()
        self.table = table
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
            sign = ','.join(nargs*'?')
            self.cmd = 'insert into %s values (%s)' % (self.table, sign)
        if self.output:
            print('Committing chunk %d (%d)' % (self.i,len(self.items)))
        self.cur.executemany(self.cmd,self.items)
        self.con.commit()
        self.items = []

