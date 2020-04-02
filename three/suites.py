from thinkbayes import Suite

class Dice(Suite):
    """
    >>> pmf = Dice([4, 6, 8, 12, 20])
    >>> pmf.Update(6)
    >>> pmf.Print()

    4 0.0
    6 0.3921568627450981
    8 0.29411764705882354
    12 0.19607843137254904
    20 0.11764705882352944

    """
    def Likelihood(self, data, hypo):
        if data > hypo:
            return 0
        else:
            return 1.0 / hypo


class Locomotive(Suite):
    """
    >>> pmf = Locomotive(list(range(1, 1001)))
    >>> pmf.Update(60)
    """
    def Likelihood(self, data, hypo):
        if data > hypo:
            return 0
        else:
            return 1.0 / hypo


class LocomotivePowerLawPrior(Locomotive):

    def __init__(self, hypos, alpha=1):
        # Parent class PMF (really _DictWrapper) __init__ with no params
        # initializes without setting any hypotheses.
        super().__init__(self)

        # This is the part we are overwriting (setting the hypotheses with
        # power law prior, rather than uniform distribution)
        for hypo in hypos:
            self.Set(hypo, hypo**(-alpha))
        self.Normalize()


def Percentile(pmf, percentage):
    p = percentage / 100.0
    total = 0
    for val, prob in pmf.Items():
        total += prob
        if total >= p:
            return val
