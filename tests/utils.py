import base64


def img_to_base64(path):
    """Converts an img to a base64 encoded string."""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()