
def time_diff(time_vals):
    time1 = time_vals
    time2 = time_vals.shift(1)
    delta = time1 - time2
    return delta