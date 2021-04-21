config = { "x":1, "y":2, "z":3, "p":4}


def config_pass_sum(m, config):
    config_sum(**config)

def config_sum(x,y,**kwargs):
    print(x + y)
    print(kwargs)


config_pass_sum(0,config)