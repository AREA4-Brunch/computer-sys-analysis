class SingletonMeta(type):
    _instances = dict()

    def __call__(cls, *args, **kwargs):
        print 
        if cls not in SingletonMeta._instances:
            cls_instance = super().__call__(*args, **kwargs)
            SingletonMeta._instances[cls] = cls_instance
        return SingletonMeta._instances[cls]
