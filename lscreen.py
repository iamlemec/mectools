#!/usr/bin/env python3

# logging screen helper

import os
from subprocess import run
import click

# runtime data location
rundir = os.path.join(os.evn['XDG_RUNTIME_DIR'], 'iscreen')
if not os.path.isdir(rundir):
    os.mkdir(rundir)
screens = os.listdir(rundir)

@click.command()
@click.argument('name', help='name of screen')
@click.option('--dir', default=None, help='directory to run in')
@click.argument('cmd', nargs=-1, help='command to run')
def create(name, cmd, dir):
    if name in screens:
        ret = run('screen -S %s -Q select . &> /dev/null' % name, shell=True)
        if ret != 0:
            print('Removing old logfile')
            slog = os.path.join(rundir, 'screenlog.0')
            os.remove(slog)
        else:
            print('Screen with that name already exists.')
            return
    cwd = os.getcwd()
    if dir:
        os.chdir(dir)
    run(['screen', '-L', '-S', name, '-d', '-m'] + cmd)
    if dir:
        os.chdir(cwd)
    log = os.path.join(dir or cwd, 'screenlog.0')
    os.symlink(log, rundir)

@click.command()
def list():
    print(screens)

# ensure elltwo "python3 server.py --path=/media/Liquid/Dropbox/elledit --ip=0.0.0.0 --port=8500 --auth=auth/auth.txt --local-libs" "/media/Liquid/work/elltwo"
# ensure kkedit "python3 server.py --path=/media/Liquid/Dropbox/kkedit --ip=0.0.0.0 --port=8501 --auth=auth/kkauth.txt --local-libs" "/media/Liquid/work/elltwo"
# ensure tidbit "python3 editor.py tidbit.db" "/media/Liquid/work/tidbit"
# ensure wikinav "python3 wikinav.py history.db" "/media/Liquid/work/mnemonic/server"
# ensure jupyter "jupyter notebook" "/media/Liquid/work/notebooks"
# ensure mogbot "python3 twitter_mogbot.py" "/media/Liquid/work/mogbot"
