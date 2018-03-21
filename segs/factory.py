from segs.fcn import fcn_net


def get_net(name):
    if 'fcn' in name:
        return fcn_net(mode=name.lstrip('fcn'))