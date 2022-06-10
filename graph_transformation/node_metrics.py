class NodeMetrics:
    def __init__(self,
                 hood_infection,
                 alpha_numerator,
                 alpha_denominator):
        self.eta = self._calculate_eta(hood_infection, alpha_denominator)
        self.alpha = self._calculate_alpha(alpha_numerator, alpha_denominator)

    @classmethod
    def _calculate_eta(cls, h_inf, a_den):
        return [d and n / d or 0 for n, d in zip(h_inf, a_den)]

    @classmethod
    def _calculate_alpha(cls, a_num, a_den):
        return [d and n / d or 0 for n, d in zip(a_num, a_den)]