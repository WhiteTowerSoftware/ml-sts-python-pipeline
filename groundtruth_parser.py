 
from io import BytesIO
import base64
import numpy as np


def decode_capture_out_put(data: str) -> np.array:
    bytes_data = base64.decodebytes(data.encode('utf-8'))
    stream = BytesIO(bytes_data)
    return np.load(stream, allow_pickle=True)



# Testing

example_data = {
        "endpointInput": {
            "observedContentType": "application/x-npy",
            "mode": "INPUT",
            "data": "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDIzKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIArmXIqryr7LP+ZciqvKvss/3AcgtYmT7D/cByC1iZPsP9wHILWJk+w/uRluwOeHYT+Y3ZOHhVroPx2s/3OYL7c/fjUHCObooT8AAAAAAAAAAGEyVTAqqUM/wOyePCzUij8w+grSjEWDPyKJXkax3JI/MPoK0oxFgz/mXIqryr7LPyKJXkax3JI/MPoK0oxFgz/MJYbydMrRPyKJXkax3JI/IoleRrHckj8AAAAAAADwPx2s/3OYL7c/",
            "encoding": "BASE64"
        },
        "endpointOutput": {
            "observedContentType": "application/x-npy",
            "mode": "OUTPUT",
            "data": "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAADwPw==",
            "encoding": "BASE64"
        }
}


if __name__ == "__main__":

    d1 = example_data["endpointInput"]['data']
    print("endpointInput = {}".format(decode_capture_out_put(d1)))