


def is_number(s):
    try:
        float(s)
        return True
    except TypeError:
        return False
    except ValueError:
        return False
