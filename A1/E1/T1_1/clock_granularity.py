import numpy as np
import time
import timeit


def checktick(timestampFunction):
    M = 200
    timesfound = np.empty((M,))
    for i in range(M):
        t1 = timestampFunction()  # get timestamp from timer
        t2 = timestampFunction()  # get timestamp from timer
        while (
            t2 - t1
        ) < 1e-16:  # if zero then we are below clock granularity, retake timing
            t2 = timestampFunction()  # get timestamp from timer
        t1 = t2  # this is outside the loop
        timesfound[i] = t1  # record the time stamp
    minDelta = 1000000
    Delta = np.diff(timesfound)  # it should be cast to int only when needed
    minDelta = Delta.min()
    return minDelta


def main():
    print(checktick(time.time))
    print(checktick(timeit.default_timer))
    print(checktick(time.time_ns))


if __name__ == "__main__":
    main()
