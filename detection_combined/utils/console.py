
from rich.console import Console

class ConsoleSingleton(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ConsoleSingleton, cls).__new__(cls)
            return cls.instance
   
    def __init__(self) -> None:
        self.console = Console()
   
    def get_console(self) -> None:
        return self.console


