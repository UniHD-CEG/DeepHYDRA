import io
import pickle

# As described in https://stackoverflow.com/a/53327348

class SPOTUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "spot":
            renamed_module = "utils.spot"

        return super(SPOTUnpickler, self).find_class(renamed_module, name)


def load(file_obj):
    return SPOTUnpickler(file_obj).load()


def loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return load(file_obj)