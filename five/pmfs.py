from thinkbayes import PMF


class Die(PMF):
    def __init__(self, sides):
        PMF.__init__(self)
        for x in range(1, sides + 1):
            self.Set(x, 1)
        self.Normalize()    