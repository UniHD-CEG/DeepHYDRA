from beauty import Beauty

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class BeautySingleton(metaclass=Singleton):   
    def __init__(self,
                    server: str = 'https://atlasop.cern.ch') -> None:
        
        self._server = server
        self._beauty_instance = Beauty(server=self._server)
   
    def instance(self) -> None:
        return self._beauty_instance


