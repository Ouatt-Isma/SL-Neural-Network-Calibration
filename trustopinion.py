import numpy as np
from fractions import Fraction


class TrustOpinion:
    """
    A Subjective Logic (SL) opinion over a binary proposition, represented as
    the quadruple (belief, disbelief, uncertainty, base_rate) satisfying
    belief + disbelief + uncertainty == 1 and all components in [0, 1].

    Attributes
    ----------
    t : float  — belief mass
    d : float  — disbelief mass
    u : float  — uncertainty mass
    a : float  — base rate (prior probability), default 0.5
    """

    def __init__(self, trust_mass, distrust_mass, untrust_mass, base_rate=0.5):
        assert trust_mass >= 0 and distrust_mass >= 0 and untrust_mass >= 0
        assert round(trust_mass + distrust_mass + untrust_mass, 10) == 1
        assert 0 <= base_rate <= 1
        self.t = trust_mass
        self.d = distrust_mass
        self.u = untrust_mass
        self.a = base_rate

    # ── Constructors ──────────────────────────────────────────────────────────

    def ev2tdu(pos_ev: int, neg_ev: int, W=2) -> "TrustOpinion":
        """Build an opinion from positive and negative evidence counts."""
        b = pos_ev / (pos_ev + neg_ev + W)
        d = neg_ev / (pos_ev + neg_ev + W)
        u = W / (pos_ev + neg_ev + W)
        return TrustOpinion(b, d, u)

    def vacuous() -> "TrustOpinion":
        """Return a vacuous opinion (0, 0, 1)."""
        return TrustOpinion(0, 0, 1)

    def ftrust() -> "TrustOpinion":
        """Return a fully trusted opinion (1, 0, 0)."""
        return TrustOpinion(1, 0, 0)

    def dtrust() -> "TrustOpinion":
        """Return a fully distrusted opinion (0, 1, 0)."""
        return TrustOpinion(0, 1, 0)

    def random(n=10) -> "TrustOpinion":
        """Return a random opinion sampled from integer triplets in [1, n)."""
        a, b, c = np.random.randint(1, n, 3)
        t = a / (a + b + c)
        d = b / (a + b + c)
        u = c / (a + b + c)
        return TrustOpinion(t, d, u)

    def random_matrix(n, m):
        """Return an (n, m) NumPy array of random opinions."""
        return np.array([[TrustOpinion.random() for _ in range(m)]
                         for _ in range(n)], dtype=TrustOpinion)

    def fill(shape, method="random", value: "TrustOpinion" = None):
        """
        Create a (rows, cols) array of opinions.

        Parameters
        ----------
        shape  : tuple of length 2
        method : 'random' | 'trust' | 'one' | 'distrust' | 'vacuous'
        value  : if set, every cell is filled with this TrustOpinion instance
        """
        if len(shape) != 2:
            raise ValueError("shape must be a 2-tuple")

        res = np.empty(shape=shape, dtype=TrustOpinion)

        if value is not None:
            if not isinstance(value, TrustOpinion):
                raise ValueError("value must be a TrustOpinion instance")
            for i in range(shape[0]):
                for j in range(shape[1]):
                    res[i][j] = value
            return res

        if method in ("trust", "one"):
            factory = TrustOpinion.ftrust
        elif method == "distrust":
            factory = TrustOpinion.dtrust
        elif method == "vacuous":
            factory = TrustOpinion.vacuous
        elif method == "random":
            factory = TrustOpinion.random
        else:
            raise ValueError(f"Unsupported fill method: {method!r}")

        for i in range(shape[0]):
            for j in range(shape[1]):
                res[i][j] = factory()
        return res

    # ── Derived quantities ────────────────────────────────────────────────────

    def projected_prob(self, frac=False):
        """Return the projected probability P = b + a·u."""
        if frac:
            return Fraction(self.t + self.a * self.u).limit_denominator()
        return round(self.t + self.a * self.u, 3)

    # ── Display ───────────────────────────────────────────────────────────────

    def print(self, frac=False):
        """Return the opinion as a formatted string '(b, d, u, a)'."""
        if frac:
            return "({},{},{},{})".format(
                Fraction(self.t).limit_denominator(),
                Fraction(self.d).limit_denominator(),
                Fraction(self.u).limit_denominator(),
                Fraction(self.a).limit_denominator(),
            )
        return "({},{},{},{})".format(
            round(self.t, 3), round(self.d, 3),
            round(self.u, 3), round(self.a, 3),
        )

    def __str__(self):
        return self.print()

    def __repr__(self):
        return self.print()

    # ── Fusion operators ──────────────────────────────────────────────────────

    def binMult(self, op2: "TrustOpinion") -> "TrustOpinion":
        """Binomial multiplication of two opinions."""
        if not isinstance(op2, TrustOpinion):
            raise ValueError("op2 must be a TrustOpinion")
        t = (self.t * op2.t
             + ((1 - self.a) * op2.a * self.t * op2.u
                + (1 - op2.a) * self.a * op2.t * self.u)
             / (1 - self.a * op2.a))
        d = self.d + op2.d - self.d * op2.d
        u = (self.u * op2.u
             + ((1 - self.a) * op2.t * self.u
                + (1 - op2.a) * self.t * op2.u)
             / (1 - self.a * op2.a))
        a = self.a * op2.a
        t = round(t, 20)
        d = round(d, 20)
        u = 1 - (t + d)
        a = round(a, 20)
        return TrustOpinion(t, d, u, a)

    def binomialMultiplication(op1: "TrustOpinion",
                                op2: "TrustOpinion") -> "TrustOpinion":
        """Static alias for op1.binMult(op2)."""
        if not isinstance(op1, TrustOpinion):
            raise ValueError("op1 must be a TrustOpinion")
        return op1.binMult(op2)

    def averaging_belief_fusion(b_A, u_A, a_A, b_B, u_B, a_B):
        """
        Averaging belief fusion for two sources A and B.

        Parameters
        ----------
        b_A, u_A, a_A : belief, uncertainty, base rate for source A
        b_B, u_B, a_B : belief, uncertainty, base rate for source B

        Returns
        -------
        (b_fused, u_fused, a_fused)
        """
        if u_A != 0 or u_B != 0:
            b_fused = (b_A * u_B + b_B * u_A) / (u_A + u_B)
            u_fused = (2 * u_A * u_B) / (u_A + u_B)
            a_fused = (a_A + a_B) / 2
        else:
            b_fused = 0.5 * (b_A + b_B)
            u_fused = 0
            a_fused = (a_A + a_B) / 2
        return b_fused, u_fused, a_fused

    def weighted_belief_fusion(b_A, u_A, a_A, b_B, u_B, a_B):
        """
        Weighted belief fusion for two sources A and B.

        Parameters
        ----------
        b_A, u_A, a_A : belief, uncertainty, base rate for source A
        b_B, u_B, a_B : belief, uncertainty, base rate for source B

        Returns
        -------
        (b_fused, u_fused, a_fused)
        """
        if (u_A != 0 or u_B != 0) and (u_A != 1 or u_B != 1):
            b_fused = (b_A * (1 - u_A) * u_B + b_B * (1 - u_B) * u_A) / (u_A + u_B)
            u_fused = ((2 - u_A - u_B) * u_A * u_B) / (u_A + u_B - 2 * u_A * u_B)
            a_fused = (a_A + a_B) / 2
        elif u_A == 0 and u_B == 0:
            b_fused = 0.5 * (b_A + b_B)
            u_fused = 0
            a_fused = (a_A + a_B) / 2
        elif u_A == 1 and u_B == 1:
            b_fused = 0
            u_fused = 1
            a_fused = (a_A + a_B) / 2
        else:
            raise ValueError("Unhandled uncertainty case in weighted_belief_fusion")
        return b_fused, u_fused, a_fused

    def avFuse(op1: "TrustOpinion", op2: "TrustOpinion") -> "TrustOpinion":
        """Averaging belief fusion of two opinions."""
        b_A, u_A, a_A = op1.t, op1.u, op1.a
        b_B, u_B, a_B = op2.t, op2.u, op2.a
        b, u, a = TrustOpinion.averaging_belief_fusion(b_A, u_A, a_A, b_B, u_B, a_B)
        return TrustOpinion(b, 1 - (b + u), u, a)

    def weigFuse(op1: "TrustOpinion", op2: "TrustOpinion") -> "TrustOpinion":
        """Weighted belief fusion of two opinions."""
        b_A, u_A, a_A = op1.t, op1.u, op1.a
        b_B, u_B, a_B = op2.t, op2.u, op2.a
        b, u, a = TrustOpinion.weighted_belief_fusion(b_A, u_A, a_A, b_B, u_B, a_B)
        return TrustOpinion(b, 1 - (b + u), u, a)

    def cumFuse(op1: "TrustOpinion", op2: "TrustOpinion") -> "TrustOpinion":
        """Cumulative belief fusion of two opinions."""
        b1, d1, u1, a1 = op1.t, op1.d, op1.u, op1.a
        b2, d2, u2, a2 = op2.t, op2.d, op2.u, op2.a

        if b1 != 0 or b2 != 0:
            b = (b1 * u2 + b2 * u1) / (u1 + u2 - u1 * u2)
            u = u1 * u2 / (u1 + u2 - u1 * u2)
            if u1 != 1 or u2 != 1:
                a = (a1 * u2 + a2 * u1 - (a1 + a2) * u1 * u2) / (u1 + u2 - 2 * u1 * u2)
            else:
                a = (a1 + a2) / 2
        else:
            b = 0.5 * (b1 + b2)
            u = 0
            a = 0.5 * (a1 + a2)

        d = 1 - u - b
        b = round(b, 2)
        d = round(d, 2)
        u = round(u, 2)
        a = round(a, 2)
        if b + d + u != 1:
            u = 1 - (b + d)
        return TrustOpinion(b, d, u, a)

    def deduction(op_x: "TrustOpinion",
                  op_y_given_x: "TrustOpinion",
                  op_y_given_not_x: "TrustOpinion") -> "TrustOpinion":
        """
        Subjective logic deduction (Jøsang, §9).

        Computes the marginal opinion on Y from:
          - op_x              : opinion on X
          - op_y_given_x      : conditional opinion on Y | X
          - op_y_given_not_x  : conditional opinion on Y | ¬X
        """
        if op_x.t == 1:
            return op_y_given_x
        if op_x.d == 1:
            return op_y_given_not_x

        ax, bx, dx, ux = op_x.a, op_x.t, op_x.d, op_x.u
        ex = op_x.projected_prob()

        b0, d0, u0 = op_y_given_x.t, op_y_given_x.d, op_y_given_x.u
        b1, d1, u1 = op_y_given_not_x.t, op_y_given_not_x.d, op_y_given_not_x.u

        ay = (ax * b0 + (1 - ax) * b1) / (1 - ax * u0 - (1 - ax) * u0)

        bIy = bx * b0 + dx * b1 + ux * (b0 * ax + b1 * (1 - ax))
        dIy = bx * d0 + dx * d1 + ux * (d0 * ax + d1 * (1 - ax))
        uIy = bx * u0 + dx * u1 + ux * (u0 * ax + u1 * (1 - ax))
        Pyvacuousx = b0 * ax + b1 * (1 - ax) + ay * (u0 * ax + u1 * (1 - ax))

        K = 0
        if ((b0 > b1) and (d0 > d1)) or ((b0 <= b1) and (d0 <= d1)):
            K = 0
        elif (b0 > b1) and (d0 <= d1):
            if Pyvacuousx <= b1 + ay * (1 - b1 - d0):
                K = (ax * ux * (bIy - b1) / (ay * ex) if ex <= ax
                     else ax * ux * (dIy - d0) * (b0 - b1)
                     / ((dx + (1 - ax) * ux) * ay * (d1 - d0)))
            else:
                K = ((1 - ax) * ux * (bIy - b1) * (d1 - d0) / (ex * (1 - ay) * (b0 - b1))
                     if ex <= ax
                     else (1 - ax) * ux * (dIy - d0)
                     / ((1 - ay) * (dx + (1 - ax) * ux)))
        else:
            if Pyvacuousx <= b1 + ay * (1 - b1 - d0):
                K = ((1 - ax) * ux * (dIy - d1) * (b1 - b0) / (ex * ay * (d0 - d1))
                     if ex <= ax
                     else (1 - ax) * ux * (bIy - b0) / (ay * (dx + (1 - ax) * ux)))
            else:
                K = (ax * ux * (dIy - d1) / (ex * (1 - ay)) if ex <= ax
                     else ax * ux * (bIy - b0) * (d0 - d1)
                     / ((1 - ay) * (b1 - b0) * (dx + (1 - ax) * ux)))

        if not isinstance(K, float) or K != K:  # guard against NaN
            K = 0

        by = bIy - ay * K
        dy = dIy - (1 - ay) * K
        uy = uIy + K
        return TrustOpinion(by, dy, uy, ay)

    # ── Operator overloads ────────────────────────────────────────────────────

    def __add__(self, other):
        return self.binMult(other)

    def __sub__(self, other):
        return self.binMult(other)

    def __mul__(self, other):
        if isinstance(other, TrustOpinion):
            return self.binMult(other)
        return other.__mul__(self)
