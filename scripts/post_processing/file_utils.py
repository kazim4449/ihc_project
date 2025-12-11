import re

def natural_sort_key(text):
    """
    Generates a key for natural sorting of strings containing numbers.

    Example:
        "image_2.jpg" < "image_10.jpg" lexicographically,
        but natural_sort_key will make it sort correctly.
    """
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', text)]


def natural_sort(iterable):
    """
    Returns a naturally sorted list of the given iterable of strings.

    Example:
        input: ['image_2.jpg', 'image_10.jpg']
        output: ['image_2.jpg', 'image_10.jpg']
    """
    return sorted(iterable, key=natural_sort_key)