from beauty import Beauty

from .singleton import Singleton


class BeautySingleton(metaclass=Singleton):   
    def __init__(self,
                    server: str = 'https://atlasop.cern.ch') -> None:
        
        self._server = server
        self._beauty_instance = Beauty(server=self._server)
   
    def instance(self) -> None:
        return self._beauty_instance


