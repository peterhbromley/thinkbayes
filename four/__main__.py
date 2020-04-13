import numpy as np
from matplotlib import pyplot as plt
from .suites import Euro, EuroNoisy, EuroTrianglePrior
from thinkbayes import Beta, CredibleInterval, Percentile
from colour import Color


HEADS = 'H'
TAILS = 'T'

def noisy_euro(wrong_probs):
    hypos = list(range(101))
    data = HEADS * 140 + TAILS * 110

    colors = list(
        Color(rgb=(.68, .85, .90)).range_to(Color(rgb=(0, 0, 1)),
                                            len(wrong_probs))
    )

    for i, prob in enumerate(wrong_probs):
        euro_noisy = EuroNoisy(hypos, wrong_prob=prob)
        
        for d in data:
            euro_noisy.Update(d)

        posterior_probs = euro_noisy.Probs(euro_noisy.Values())

        if i == 0 or i == len(wrong_probs) - 1:
            plt.plot(hypos, posterior_probs, c=colors[i].rgb, label=f'noise prob {prob:.2f}')
        else:
            plt.plot(hypos, posterior_probs, c=colors[i].rgb)


        print(f'Mean for noise prob {prob}: {euro_noisy.Mean()}')
    plt.legend()
    plt.savefig('four/euro_noisy.png')
    

def main():
    """
    Euro problem
    
    Description
    -----------
    A Euro is spun 250 times, comes up heads 140 times and tails 110 times.
    Do these data give evidence that the coin is biased?

    Components
    ----------
    - Hypotheses:
        `H_0, H_1, ..., H_100`
        where `H_x` is the hypothesis that prob of heads is `x` percent.

    - Likelihood P(D|H):
        Given H_x is true, P(heads) = x/100, P(tails) = 1 - x/100.

    - Prior: Uniform for the first run, Triangle for the second run.
    """
    print('')
    print('')
    print('')

    print('Euro problem: ')
    hypos = list(range(101))
    euro = Euro(hypos)
    euro_tri = EuroTrianglePrior(hypos)
    data = HEADS * 140 + TAILS * 110
    
    for d in data:
        euro.Update(d)
        euro_tri.Update(d)
    
    print('Plotting posterior distribution (uniform prior)')
    plt.plot(hypos, euro.Probs(euro.Values()), label='uniform')
    print(f'Mean: {euro.Mean()}')
    print(f'Median: {Percentile(euro, 50)}')
    print(f'Credible Interval: {CredibleInterval(euro, 90)}')
    print(f'Maximum Likelihood: {euro.MaximumLikelihood()}')
    print('')

    print('Plotting posterior distribution (triangle prior)')
    plt.plot(hypos, euro_tri.Probs(euro_tri.Values()), label='triangle')
    print('')

    plt.legend()
    plt.savefig('four/euro.png')
    plt.close()
    print('Saving posterior plot as euro.png')
    print('')

    print('Using conjugate prior beta distribution')
    alpha = 1
    beta = 1
    print(f'alpha: {alpha}, beta: {beta}')
    euro_beta = Beta(alpha=alpha, beta=beta)
    euro_beta.Update((140, 110))
    print(f'Mean: {euro_beta.Mean()}')
    print('')

    print('Exercise 4.1: ')

    
    print(f'Plotting posterior distribution (noisy euro problem)')
    wrong_probs = np.arange(0, 1.05j, 0.05)
    noisy_euro(wrong_probs)
    

    print('')
    print('')
    print('')

if __name__ == '__main__':
    main()
