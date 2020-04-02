from matplotlib import pyplot as plt
from .suites import Dice, Locomotive, LocomotivePowerLawPrior, Percentile

def main():
    """
    DESCRIPTION OF DICE PROBLEM
    """
    print('Dice Problem: ')
    hypotheses = [4, 6, 8, 12, 20]
    data = 6
    pmf = Dice(hypotheses)
    pmf.Update(data)
    pmf.Print()
    print('')

    """
    DESCRIPTION OF LOCOMOTIVE PROBLEM
    """
    print('Locomotive Problem: ')
    hypotheses = list(range(1, 1001))
    data = 60
    pmf = Locomotive(hypotheses)
    pmf.Update(data)
    plt.plot(pmf.Values(), pmf.Probs(pmf.Values()), label='uniform prior')
    print(f'Mean: {pmf.Mean()}')
    print('')

    print('Locomotive Problem, with power law prior: ')

    pmf = LocomotivePowerLawPrior(list(range(1, 1001)))
    pmf.Update(60)
    plt.plot(pmf.Values(), pmf.Probs(pmf.Values()), label='power law prior')
    plt.legend()
    plt.savefig('three/locomotive.png')


    # Locomotive problem with power law prior, 3 updates
    hypotheses = [
        list(range(1, 501)),
        list(range(1, 1001)),
        list(range(1, 2001)),
    ]
    data = [30, 60, 90]
    for hypos in hypotheses:
        pmf = LocomotivePowerLawPrior(hypos)
        for d in data:
            pmf.Update(d)
        print(f'Upper bound: {hypos[-1]}, Mean: {pmf.Mean()}')

    interval = Percentile(pmf, 5), Percentile(pmf, 95)
    print(f'90 percent credible interval: {interval}')
    print('')

    print('Convert to a CDF')
    cdf = pmf.MakeCDF()
    interval = cdf.Percentile(5), cdf.Percentile(95)
    print(f'90 percent credible interval (using CDF): {interval}')


if __name__ == '__main__':
    main()
