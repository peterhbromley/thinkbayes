from .pmfs import Die
from matplotlib import pyplot as plt
from thinkbayes import SampleSum, PMF

def main():
    print('Distribution of sum of 3 6-sided dice: ')
    """
    Calculated by simulation.
        Create three 6 sided die pmf, take random samples from them
        and add them up.
    """
    die = Die(6)
    all_dice = [die, die, die]
    # Generate distribution of sum by sampling 1000 rolls
    sum_distribution = SampleSum(all_dice, 1000)
    print(f'Mean (simulation): {sum_distribution.Mean()}')
    hypos, probs = sum_distribution.XY()
    plt.plot(hypos, probs, linestyle='-', label='sampled')

    """
    Calculated by enumeration.
        Enumerate all pairs of values, compute the sum and probability
        of each pair.
    """
    sum_distribution = die + die + die
    hypos = sorted(sum_distribution.Values())
    probs = sum_distribution.Probs(hypos)
    plt.plot(hypos, probs, label='exact')
    plt.legend()
    plt.savefig('five/three_dice.png')

    """
    Calculate max distr by converting to CDF.
        pmf.Max(k), where k is the number of selections (in our case, we are
        looking for the max of 6 attributes, so k=6).
    """
    best_attribute_cdf = sum_distribution.Max(6)
    best_attribute_pmf = best_attribute_cdf.MakePMF()
    xs, ys = best_attribute_pmf.XY()
    plt.plot(xs, ys, label='max')
    plt.legend()
    plt.savefig('five/three_dice_max.png')
    plt.close()


    """
    Mixture problem:

    Roll a die from the following set:
        sides | n
       -----------
            4 | 5
            6 | 4
            8 | 3
           12 | 2
           20 | 1
    Distribution of roll?
    """
    pmf_dice = PMF()
    pmf_dice.Set(Die(4), 5)
    pmf_dice.Set(Die(6), 4)
    pmf_dice.Set(Die(8), 3)
    pmf_dice.Set(Die(12), 2)
    pmf_dice.Set(Die(20), 1)
    pmf_dice.Normalize()

    mix = PMF()
    for die, weight in pmf_dice.Items():
        for outcome, prob in die.Items():
            mix.Incr(outcome, weight*prob)

    xs, ys = mix.XY()
    plt.bar(xs, ys)
    plt.savefig('five/mixture.png')
    plt.close()
    

    

if __name__ == '__main__':
    main()