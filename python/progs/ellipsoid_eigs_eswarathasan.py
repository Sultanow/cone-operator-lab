import math

def _l1_eigs(alpha, beta, gamma, eps=1.0):
    """Eigenwerte für l = 1: Formeln (38)–(40)."""
    fac = -4.0 / 5.0 * eps
    lam0 = 2.0  # l(l+1)
    return [
        lam0 + fac * (alpha + beta + 3.0 * gamma),  # u0 = v0
        lam0 + fac * (3.0 * alpha + beta + gamma),  # u0 = v1
        lam0 + fac * (alpha + 3.0 * beta + gamma),  # u0 = w1
    ]


def _l2_eigs(alpha, beta, gamma, eps=1.0):
    """Eigenwerte für l = 2: Formel (41)."""
    lam0 = 6.0  # l(l+1)
    s = alpha + beta + gamma
    q = alpha**2 + beta**2 + gamma**2 - alpha*beta - alpha*gamma - beta*gamma
    res = []

    # 2x2-Block (M_cos,e): zwei Eigenwerte
    base = lam0 - 4.0 * eps * s
    delta = (16.0 / 7.0) * eps * math.sqrt(q)
    res.append(base + delta)
    res.append(base - delta)

    # drei 1x1-Blöcke
    res.append(lam0 - (12.0 / 7.0) * eps * (3.0 * alpha + beta + 3.0 * gamma))
    res.append(lam0 - (12.0 / 7.0) * eps * (3.0 * alpha + 3.0 * beta + gamma))
    res.append(lam0 - (12.0 / 7.0) * eps * (alpha + 3.0 * beta + 3.0 * gamma))
    return res


def _l3_eigs(alpha, beta, gamma, eps=1.0):
    """
    Eigenwerte für l = 3:
    im Paper explizit unterhalb von (41) angegeben (M_cos,e, M_cos,o, M_sin,e, M_sin,o).
    """
    lam0 = 12.0  # l(l+1)
    res = []

    # M_cos,e: 2x2-Block
    base = eps * (-104.0 * alpha / 15.0 - 104.0 * beta / 15.0 - 152.0 * gamma / 15.0)
    q = 4.0 * alpha**2 + 4.0 * beta**2 - 7.0 * alpha * beta - alpha * gamma - beta * gamma + gamma**2
    delta = eps * 32.0 / 15.0 * math.sqrt(q)
    res.append(lam0 + base + delta)
    res.append(lam0 + base - delta)

    # M_cos,o: 2x2-Block
    base = eps * (-104.0 * gamma / 15.0 - 104.0 * beta / 15.0 - 152.0 * alpha / 15.0)
    q = 4.0 * gamma**2 + 4.0 * beta**2 - 7.0 * gamma * beta - gamma * alpha - beta * alpha + alpha**2
    delta = eps * 32.0 / 15.0 * math.sqrt(q)
    res.append(lam0 + base + delta)
    res.append(lam0 + base - delta)

    # M_sin,e: 1x1
    base = eps * (-8.0 * alpha - 8.0 * beta - 8.0 * gamma)
    res.append(lam0 + base)

    # M_sin,o: 2x2-Block
    base = eps * (-104.0 * gamma / 15.0 - 104.0 * alpha / 15.0 - 152.0 * beta / 15.0)
    q = 4.0 * gamma**2 + 4.0 * alpha**2 - 7.0 * gamma * alpha - gamma * beta - alpha * beta + beta**2
    delta = eps * 32.0 / 15.0 * math.sqrt(q)
    res.append(lam0 + base + delta)
    res.append(lam0 + base - delta)

    # ergibt 7 Eigenwerte (2l+1)
    return res


def ellipsoid_surface_eigs_triaxial(a, b, c, eps=1.0):
    """
    Approximiert die ersten Eigenwerte des Laplace-Beltrami-Operators
    auf dem triaxialen Ellipsoid E_{a,b,c} mit den Formeln aus dem Paper.

    a, b, c  : Halbachsen des Ellipsoids
    eps      : Skalenparameter der Störung (in den Formeln a=1+α eps etc.).
               Für a≈b≈c≈1 nimm eps≈1 und α=a-1, β=b-1, γ=c-1.
    """
    # Parametrisierung wie im Paper: a = 1 + α eps, usw.
    alpha = (a - 1.0) / eps
    beta  = (b - 1.0) / eps
    gamma = (c - 1.0) / eps

    # l=0: konstante Funktion, Eigenwert bleibt 0 (Laplace-Beltrami)
    eigenvalues = [0.0]

    # l=1, 2, 3 mit den expliziten Formeln
    eigenvalues.extend(_l1_eigs(alpha, beta, gamma, eps=eps))
    eigenvalues.extend(_l2_eigs(alpha, beta, gamma, eps=eps))
    eigenvalues.extend(_l3_eigs(alpha, beta, gamma, eps=eps))

    # sortiert zurückgeben
    eigenvalues_sorted = sorted(eigenvalues)
    return eigenvalues_sorted


if __name__ == "__main__":
    # Dein Beispiel: a=1, b=1.5, c=2.3
    a, b, c = 1.0, 1.5, 2.3
    eps = 1.0  # wir interpretieren α=a-1, β=b-1, γ=c-1 als erste Ordnung

    lambdas = ellipsoid_surface_eigs_triaxial(a, b, c, eps=eps)
    print("Approx. erste Eigenwerte für E_{1, 1.5, 2.3} (Laplace-Beltrami, 1. Ordnung):")
    for i, lam in enumerate(lambdas):
        print(f"{i:2d}: {lam:.6f}")
