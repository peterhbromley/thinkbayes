from thinkbayes import Suite

class Cookie(Suite):
    mixes = {
        'b1': {
            'vanilla': 0.75,
            'chocolate': 0.25,
        },
        'b2': {
            'vanilla': 0.5,
            'chocolate': 0.5,
        },
    }
    def Likelihood(self, data, hypo):
        mix = self.mixes[hypo]
        likelihood = mix[data]
        return likelihood


class MontyHall(Suite):
    mixes = {
        'A': dict(A=0, B=0.5, C=0.5),
        'B': dict(A=0, B=0,   C=1),
        'C': dict(A=0, B=1,   C=0),
    }       
    def Likelihood(self, data, hypo):
        mix = self.mixes[hypo]
        likelihood = mix[data]
        return likelihood


class Bowl:
    def __init__(self, cookies):
        self.cookies = cookies
    
    @property
    def total(self):
        return sum(self.cookies.values())

    def take(self, cookie):
        assert self.cookies[cookie] > 0, f'No more {cookie} cookies left'
        self.cookies[cookie] -= 1

    @property
    def probabs(self):
        return {
            c: self.cookies[c] / self.total for c in self.cookies.keys()
        }


class CookieWithoutReplacement(Suite):
    bowls = {
        'b1': Bowl(dict(vanilla=30, chocolate=10)),
        'b2': Bowl(dict(vanilla=20, chocolate=20)),
    }
    def Likelihood(self, data, hypo):
        bowl = self.bowls[hypo]
        likelihood = bowl.probabs[data]
        bowl.take(data)
        return likelihood


