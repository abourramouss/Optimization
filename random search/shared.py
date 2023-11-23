import math, random


def basinFunction(vector: list) -> float:
    a, h, k = 0.5, 2, -5
    sum = 0
    for item in vector:
        sum = sum + a * pow((item - h), 2) + k
    return sum


def basinFunction1(vector: list) -> float:
    sum_sq = 0
    for x_i in vector:
        sum_sq += x_i**2
    return sum_sq
