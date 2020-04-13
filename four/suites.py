from thinkbayes import Suite

HEADS = 'H'
TAILS = 'T'

class Euro(Suite):
    def Likelihood(self, data, hypo):
        """
        Each hypothesis represents a possible percent chance that the
        Euro comes up heads.
        """
        if data == HEADS:
            return hypo / 100.0
        else:
            return 1 - (hypo / 100.0)


class EuroTrianglePrior(Suite):
    def __init__(self, hypos):
        super().__init__(self)

        # Ascribe more probability mass to numbers closer to 50%
        for x in range(0, 51):
            self.Set(x, x)
        for x in range(51, 101):
            self.Set(x, 100-x)

        self.Normalize()

    def Likelihood(self, data, hypo):
        if data == HEADS:
            return hypo / 100.0
        else:
            return 1 - (hypo / 100.0)


class EuroOptimized(Suite):
    def Likelihood(self, data, hypo):
        """
        Encode all of the observations in `data`, so we don't have to
        iterate over all of the observations.

        Params
        ------
        data: (int, int)
            where each tuple value is number of observed heads/tails.
        hypo: int
            percent of heads for the given hypothesis.

        Example
        -------
            observations = (140, 110)
            euro.Update(observations)
        """
        p = hypo / 100.0
        heads, tails = data
        like = p**heads * (1-p)**tails
        return like


class EuroNoisy(Suite):
    def __init__(self, hypos, wrong_prob=0.1):
        super().__init__(values=hypos)

        self.wrong_prob = wrong_prob
    
    def Likelihood(self, data, hypo):
        p_heads = hypo / 100.0
        p_tails = 1 - (hypo / 100.0)
        p_correct = (1 - self.wrong_prob)
        p_incorrect = self.wrong_prob
        if data == HEADS:
            return (
                (p_heads * p_correct) + (p_tails * p_incorrect)
            )
        else:
            return (
                (p_tails * p_correct) + (p_heads * p_incorrect)
            )