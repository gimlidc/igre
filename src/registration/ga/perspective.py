import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.measure.mutual_information import mi


def fitness(pts, imgA, imgB):
    """
    Computes mutual information of "registered" images
    """
    M = cv2.getPerspectiveTransform(pts[0, :, :], pts[1, :, :])
    dst = cv2.warpPerspective(imgA, M, (imgB.shape[1], imgB.shape[0]))
    return mi(dst, imgB)


def cross(ptsA, ptsB):
    pts = np.zeros(ptsA.shape)
    for i in range(ptsA.shape[0]):
        for j in range(ptsA.shape[1]):
            alpha = np.random.uniform()
            pts[i, j, :] = alpha * ptsA[i, j, :] + (1 - alpha) * np.array(ptsB[i, j, :])
    return pts


def mutate(pts, imgA, imgB, strength=1):
    out = np.zeros(pts.shape)
    for i in range(4):
        out[0, i, 0] = np.clip(pts[0, i, 0] + np.random.normal(0, strength, 1), 0, imgA.shape[0])
        out[0, i, 1] = np.clip(pts[0, i, 1] + np.random.normal(0, strength, 1), 0, imgA.shape[1])
        out[1, i, 0] = np.clip(pts[1, i, 0] + np.random.normal(0, strength, 1), 0, imgB.shape[0])
        out[1, i, 1] = np.clip(pts[1, i, 0] + np.random.normal(0, strength, 1), 0, imgB.shape[1])
    return out


def select(fitnesses, size, elite=0):
    if elite != 0:
        surviving_elite = (-fitnesses).argsort()[:elite]
    total = sum(fitnesses)
    if total == 0:
        return np.random.randint(0, len(fitnesses), size)
    ranges = [0]
    for fitness in fitnesses:
        ranges.append(ranges[-1] + fitness)
    selection = np.random.randint(0, total, size - elite)
    if elite != 0:
        return np.concatenate((surviving_elite, np.searchsorted(ranges[1:], selection)))
    return np.searchsorted(ranges[1:], selection)


def init_population(imgA, imgB, pop_size=500):
    population = np.zeros((pop_size, 2, 4, 2), dtype=np.float32)
    for p in range(pop_size):
        for ptidx in range(4):
            population[p, 0, ptidx, 0] = np.random.randint(0, imgA.shape[0])
            population[p, 0, ptidx, 1] = np.random.randint(0, imgA.shape[1])
            population[p, 1, ptidx, 0] = np.random.randint(0, imgB.shape[0])
            population[p, 1, ptidx, 1] = np.random.randint(0, imgB.shape[1])
    return population


def optimize(imgA, imgB, iterations=25, pop_size=50, select_ratio=0.7):
    newpop = init_population(imgA, imgB, pop_size)
    for j in range(iterations):
        population = newpop
        fitnesses = np.array([fitness(pop, imgA, imgB) for pop in population])
        p2ind = select(fitnesses, int(pop_size * select_ratio), elite=1)
        newpop = np.zeros(population.shape)
        newpop[:len(p2ind), :, :, :] = population[p2ind, :, :, :]
        ind = 0
        for i in range(len(p2ind), population.shape[0] - 1, 2):
            newpop[i, :, :, :] = mutate(
                cross(population[p2ind[ind]],
                      population[p2ind[ind + 1]]),
                imgA,
                imgB,
                0.06 * j)
            newpop[i + 1, :, :, :] = mutate(
                cross(population[p2ind[ind]],
                      population[p2ind[ind + 1]]),
                imgA,
                imgB,
                0.06 * j)
            ind += 2
        newpop = newpop.astype(np.float32)
        best = (-fitnesses).argsort()[0]
        print(f"Generation {j}, highest fitness {fitnesses[best]}.")

    M = cv2.getPerspectiveTransform(population[best, 0, :, :], population[best, 1, :, :])

    plt.imshow(cv2.warpPerspective(imgA, M, (imgB.shape[0], imgB.shape[1])))
    plt.show()

    return population[best], best, M
