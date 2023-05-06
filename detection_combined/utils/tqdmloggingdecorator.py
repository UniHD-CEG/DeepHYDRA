from tqdm.contrib.logging import logging_redirect_tqdm

def tqdmloggingdecorator(func):
    def wrapper(*args, **kwargs):
        with logging_redirect_tqdm():
            return func(*args, **kwargs)
    return wrapper