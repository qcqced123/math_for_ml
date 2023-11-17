import numpy as np
from typing import Callable
from matplotlib import pyplot as plt


def volume(t):
    """ volume function """
    return (t-4)**3 / 64 + 3.3


def flow_rate(t):
    """ derivative function from volume """
    return 3*(t-4)**2 / 64


def average_flow_rate(v: Callable, t1: float, t2: float) -> float:
    """
    Calculate average flow rate, differential for delta
    Same as secant line
    Args:
        v: volume function
        t1: time stamp 1
        t2: time stamp 2
    """
    return (v(t2) - v(t1))/(t2 - t1)


def interval_flow_rate(v: Callable, t1: float, t2: float, dt: float) -> list[tuple[float, float]]:
    """
    Calculate average flow rate over interval for ascending function
    Args:
        v: volume function, ascending_volume
        t1: start time stamp
        t2: end time stamp
        dt: interval size value
    """
    return [(t, average_flow_rate(v, t, t+dt)) for t in np.arange(t1, t2, dt)]


def average_speed(v1: float, v2: float, t1: float, t2: float) -> float:
    """
    Calculate average speed, differential for delta
    Same as secant line
    Args:
        v1: velocity 1
        v2: velocity 2
        t1: time stamp 1
        t2: time stamp 2
    """
    return (v2 - v1)/(t2 - t1)


def secant_line(f: Callable, x1: float, x2: float) -> any:
    """
    secant_line: 할선, 평균 변화율 선분을 포함 하는 직선
    Calculate secant line between x1 and x2
    """
    def line(x: float) -> float:
        y1, y2 = f(x1), f(x2)
        return ((y2 - y1)/(x2 - x1)) * (x - x2)
    return line


def plot_interval_flow_rates(v: Callable, t1: float, t2: float, dt: float):
    """
    Plot interval flow rates
    Args:
        v: volume function
        t1: start time stamp
        t2: end time stamp
        dt: interval size value

    """
    series = interval_flow_rate(v, t1, t2, dt)
    times = [t for t, _ in series]
    rates = [r for _, r in series]
    plt.scatter(times, rates)
    plt.show()  # show plot


def plot_function(f, tmin, tmax, tlabel=None, xlabel=None, axes=False, **kwargs):
    """ plot some value or function ...etc from author of textbook """
    ts = np.linspace(tmin,tmax,1000)
    if tlabel:
        plt.xlabel(tlabel,fontsize=18)
    if xlabel:
        plt.ylabel(xlabel,fontsize=18)
    plt.plot(ts, [f(t) for t in ts], **kwargs)
    if axes:
        total_t = tmax-tmin
        plt.plot([tmin-total_t/10,tmax+total_t/10],[0,0],c='k', linewidth=1)
        plt.xlim(tmin-total_t/10,tmax+total_t/10)
        xmin, xmax = plt.ylim()
        plt.plot([0, 0], [xmin, xmax], c='k', linewidth=1)
        plt.ylim(xmin, xmax)


def decreasing_volume(t):
    """ decreasing volume function """
    if t < 5:
        return 10 - (t**2)/5
    else:
        return 0.2*(10-t)**2


def line_volume_function(t):
    """ line volume function """
    return 4*t + 1


def instantaneous_flow_rate(v: Callable, t: float, d: int = 6):
    """
    Calculate instantaneous flow rate, same as derivative function for volume
    Args:
        v: volume function, original function, accumulated function
        t: time stamp
        d: num digits for tolerance value
    Variables:
        tolerance: threshold for iteration, like as early stop in ML
        i: interval scaler value
    """
    tolerance, h = 10**(-d), 1
    approx = average_flow_rate(v, t-h, t+h)
    for i in range(0, 2*d):
        h = h / 10
        next_approx = average_flow_rate(v, t-h, t+h)
        if abs(next_approx - approx) < tolerance:
            return next_approx
        else:
            approx = next_approx
    raise Exception("Derivative did not coverage")


def get_flow_rate_function(v: Callable) -> Callable:
    """
    Function for return flow rate function which is getting input as volume function
    This is architecture called 'currying'
    Args:
        v: volume function
    """
    def flow_rate_function(t: float):
        return instantaneous_flow_rate(v, t)
    return flow_rate_function


# Calculate average rate of flow rate
print(f'Average Flow Rate: {average_flow_rate(volume, 4, 9)}')

# Calculate average speed
print(f'Average Speed: {average_speed(77641, 77905, 12.0, 16.5)}')

# Calculate & Visualize interval average flow rate
print(f'Interval Flow Rate: {interval_flow_rate(volume, 0, 10, 1)}')
plot_interval_flow_rates(volume, 0, 10, 1)  # ascending function
plot_interval_flow_rates(decreasing_volume, 0, 10, 1)  # descending function
plot_interval_flow_rates(line_volume_function, 0, 10, 1)  # linear function
print(instantaneous_flow_rate(volume, 1))

# Compare flow rate function: real & approximate
plot_function(flow_rate, 0, 10)
plot_function(get_flow_rate_function(volume), 0, 10)

