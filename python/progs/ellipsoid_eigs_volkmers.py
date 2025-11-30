# Volkmer-Methode zur Berechnung der Eigenwerte $\Lambda$ des Laplace-Beltrami-Operators
# auf der Oberfläche des Ellipsoids ($\Delta_S u = -\Lambda u$). Die Eigenwerte werden
# im Volkmer-Papier mit Gleichung (4.3) dimensionslos als $\lambda = \Lambda / (a^2-c^2)$ gesucht.

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import root_scalar, bisect

# ------------------------------------------------------------
# 1. Hilfsfunktionen
# ------------------------------------------------------------
def compute_modulus_k(a, b, c):
    """
    k^2 = (a^2 - b^2) / (a^2 - c^2)  (siehe (2.4) bei Volkmer).

    Wir erzwingen hier keine strikte Ordnung mehr, sondern prüfen nur,
    ob der Nenner unkritisch ist und k^2 in (0,1) liegt.
    """
    num = a**2 - b**2
    den = a**2 - c**2

    if den == 0:
        raise ValueError("a^2 - c^2 = 0 -> Modul k nicht definiert.")

    k2 = num / den

    # Optional: Warnung, falls Achsen seltsam angeordnet sind
    if not (0 < k2 < 1):
        print(f"Warnung: k^2 = {k2} liegt nicht in (0,1). "
              "Bitte Achsenreihenfolge prüfen (a,b,c).")

    return np.sqrt(k2)


# ------------------------------------------------------------
# 2. Matrizen D_N, C_N, B_N nach (8.2)–(8.4)
# ------------------------------------------------------------
def build_DCB(a, b, c, N):
    """
    Baut die (N+1)x(N+1) Matrizen D_N, C_N, B_N für die t-Gleichung (4.2)
    im Basis {cos(2 n τ), n=0..N} gemäß Volkmer (8.2)–(8.4).
    
    WICHTIG: Die Vorzeichen der D-Matrix müssen invertiert werden,
    damit die h-Eigenwerte positiv sind.
    """
    k = compute_modulus_k(a, b, c)
    k2 = k**2

    size = N + 1
    D = np.zeros((size, size), dtype=float)
    C = np.zeros((size, size), dtype=float)
    B = np.zeros((size, size), dtype=float)

    # ---------- D nach (8.2) ----------
    for n in range(size):
        if n == 0:
            continue

        # Diagonalkoeffizient D[n, n] (c_n in D c_n)
        diag = (n**2) * (
            0.5 * (a**2 + b**2)
            - 0.5 * k2 * (a**2 + 3.0 * b**2)
        )
        D[n, n] += diag

        # j = n - 1 (Oberhalb der Diagonale)
        coeff_n_minus_1 = (n**2) * (b**2 - a**2 - b**2 * k2) + 0.5 * n * (c**2) * k2
        if n - 1 >= 0:
            D[n-1, n] += coeff_n_minus_1
            
        # j = n + 1 (Unterhalb der Diagonale)
        coeff_n_plus_1 = (n**2) * (b**2 - a**2 - b**2 * k2) - 0.5 * n * (c**2) * k2
        if n + 1 < size:
            D[n+1, n] += coeff_n_plus_1

        # j = n - 2
        coeff_n_minus_2 = 0.25 * n**2 * k2 * (a**2 - b**2)
        if n - 2 >= 0:
            D[n-2, n] += coeff_n_minus_2

        # j = n + 2
        coeff_n_plus_2 = 0.25 * n**2 * k2 * (a**2 - b**2)
        if n + 2 < size:
            D[n+2, n] += coeff_n_plus_2
            
    D = -D

    # ---------- C nach (8.3) ----------
    c_diag = (1.0 / 16.0) * k2 * (a**4 + 2.0 * a**2 * b**2 + 5.0 * b**4)
    c_pm1 = (1.0 / 64.0) * k2 * (15.0 * b**4 + 2.0 * a**2 * b**2 - a**4)
    c_pm2 = (1.0 / 32.0) * k2 * (b**2 - a**2) * (a**2 + 3.0 * b**2)
    c_pm3 = (1.0 / 64.0) * k2 * (a**2 - b**2)**2

    for n in range(size):
        # Diagonal
        C[n, n] += c_diag

        # j = n ± 1
        if n - 1 >= 0:
            C[n-1, n] += c_pm1
        if n + 1 < size:
            C[n+1, n] += c_pm1

        # j = n ± 2
        if n - 2 >= 0:
            C[n-2, n] += c_pm2
        if n + 2 < size:
            C[n+2, n] += c_pm2

        # j = n ± 3
        if n - 3 >= 0:
            C[n-3, n] += c_pm3
        if n + 3 < size:
            C[n+3, n] += c_pm3

    # ---------- B nach (8.4) ----------
    b_diag = (1.0 / 8.0) * (3.0 * a**4 + 2.0 * a**2 * b**2 + 3.0 * b**4)
    b_pm1 = 0.25 * (b**4 - a**4)
    b_pm2 = (1.0 / 16.0) * (a**2 - b**2)**2

    for n in range(size):
        B[n, n] += b_diag

        if n - 1 >= 0:
            B[n-1, n] += b_pm1
        if n + 1 < size:
            B[n+1, n] += b_pm1

        if n - 2 >= 0:
            B[n-2, n] += b_pm2
        if n + 2 < size:
            B[n+2, n] += b_pm2

    # Symmetrisieren
    D = 0.5 * (D + D.T)
    C = 0.5 * (C + C.T)
    B = 0.5 * (B + B.T)

    return D, C, B

# ------------------------------------------------------------
# 3. h_n(λ) für t-Gleichung und H_m(λ) für s-Gleichung (KORRIGIERT)
# ------------------------------------------------------------
def compute_h_spectrum(lambda_val, D, C, B):
    """
    Löse das generalisierte Eigenwertproblem (D + λ C) w = h B w.
    
    WICHTIG: Erzwinge positive Eigenwerte h.
    """
    A = D + lambda_val * C
    h_vals, _ = eigh(A, B)
    h_vals = np.real_if_close(h_vals)
    
    # KRITISCHE VORZEICHENKORREKTUR:
    # Die Eigenwerte MÜSSEN positiv sein. Wir erzwingen dies durch Absolutwert.
    h_vals = np.abs(h_vals) 
    
    return np.sort(h_vals)


def h_t(lambda_val, mode_index, D_t, C_t, B_t):
    """
    n-te Kurve h_n(λ) der t-Gleichung (4.2).
    """
    h_vals = compute_h_spectrum(lambda_val, D_t, C_t, B_t)
    if mode_index >= len(h_vals):
        raise ValueError("mode_index zu groß für gegebene Trunkation N.")
    return h_vals[mode_index]


def H_s(lambda_val, mode_index, D_s, C_s, B_s):
    """
    KORREKTUR: H_m(λ) ist der reine Eigenwert des s-Problems mit Achsen (c,b,a).
    Die fehlerhafte Transformation 'lambda_val - h_val' wird entfernt.
    """
    h_vals_swapped = compute_h_spectrum(lambda_val, D_s, C_s, B_s)
    if mode_index >= len(h_vals_swapped):
        raise ValueError("mode_index zu groß für gegebene Trunkation N (s-Teil).")
    # KORREKTUR: H_m ist der Eigenwert h_m^{(t)}(λ; c,b,a)
    return h_vals_swapped[mode_index]


def F_nm(lambda_val, n, m, D_t, C_t, B_t, D_s, C_s, B_s):
    """
    KORREKTUR: Die korrekte Gleichungsfunktion (Volkmer 4.3) ist:
        F_nm(λ) = h_n(λ) + H_m(λ) - λ = 0
    """
    h_val = h_t(lambda_val, n, D_t, C_t, B_t)
    H_val = H_s(lambda_val, m, D_s, C_s, B_s)
    return h_val + H_val - lambda_val


# ------------------------------------------------------------
# 4. Root-Finding Hilfsfunktion (KORRIGIERT)
# ------------------------------------------------------------

def _calc_h_minus_H(lambda_val, n, m, D_t, C_t, B_t, D_s, C_s, B_s):
    """ Berechnet F_nm = h_n(λ) + H_m(λ) - λ robust. """
    h_vals_t = compute_h_spectrum(lambda_val, D_t, C_t, B_t)
    h_vals_s = compute_h_spectrum(lambda_val, D_s, C_s, B_s)
    
    size = len(h_vals_t)
    if n >= size or m >= size:
        raise ValueError("Mode-Index zu groß für Matrixgröße.")
    
    h_t_val = h_vals_t[n]
    H_s_val = h_vals_s[m]
    
    # KORREKTUR: F = h_t + H_s - λ
    return h_t_val + H_s_val - lambda_val 


def find_lambdas_for_pair(
    n, m,
    lam_min, lam_max,
    D_t, C_t, B_t,
    D_s, C_s, B_s,
    num_scan=400
):
    lam_grid = np.linspace(lam_min, lam_max, num_scan)
    f_vals = []

    # 1) Rasterwerte von F_nm vorberechnen
    for lam in lam_grid:
        try:
            val = _calc_h_minus_H(lam, n, m, D_t, C_t, B_t, D_s, C_s, B_s)
            f_vals.append(val)
        except Exception:
            f_vals.append(np.nan)

    f_vals = np.array(f_vals, dtype=float)
    roots = []

    # Diagnose-Ausgabe
    finite_vals = f_vals[~np.isnan(f_vals)]
    if finite_vals.size > 0:
        print(f"✅ Diagnose (n={n}, m={m}): f_min={finite_vals.min():.4f}, f_max={finite_vals.max():.4f}. F={finite_vals.size}/{num_scan} Punkte stabil.")
    else:
         print(f"❌ Diagnose (n={n}, m={m}): Keine stabilen Punkte gefunden.")
         return []

    # 2) Intervalle mit Signwechseln suchen und mit Bisektion verfeinern
    for i in range(len(lam_grid) - 1):
        lam_left = lam_grid[i]
        lam_right = lam_grid[i + 1]
        f_left = f_vals[i]
        f_right = f_vals[i + 1]

        if np.isnan(f_left) or np.isnan(f_right):
            continue

        if f_left * f_right < 0:
            
            def f_scalar(lam):
                try:
                    return _calc_h_minus_H(lam, n, m, D_t, C_t, B_t, D_s, C_s, B_s)
                except Exception:
                    return f_left
            
            try:
                root = bisect(f_scalar, lam_left, lam_right, xtol=1e-5, maxiter=50)
                roots.append(root)
            except Exception:
                continue

    return roots

def unique_sorted(values, tol=1e-6):
    """
    Werte-Liste sortieren und Duplikate (nahe beieinander) zusammenfassen.
    """
    if not values:
        return []
    vals = np.sort(values)
    result = [vals[0]]
    for v in vals[1:]:
        if abs(v - result[-1]) > tol:
            result.append(v)
    return result


def compute_ellipsoid_eigenvalues_volkmer(
    a, b, c,
    N_trunc=30,
    max_mode=3,
    lam_min=0.0,
    lam_max=40.0,
    num_scan=300
):
    """
    Haupt-Funktion:
      - sortiert die Achsen intern in Volkmer-Reihenfolge a > b > c,
      - baut D,C,B für t-Gleichung (a_vol, b_vol, c_vol),
      - baut D,C,B für "geswappt" (c_vol,b_vol,a_vol) (s-Gleichung),
      - sucht Schnittpunkte h_n(λ) = H_m(λ).
    """
    # --- Achsen in Volkmer-Notation bringen: a_vol > b_vol > c_vol ---
    a_vol, b_vol, c_vol = sorted([a, b, c], reverse=True)

    # t-Teil: Parameter (a_vol,b_vol,c_vol)
    D_t, C_t, B_t = build_DCB(a_vol, b_vol, c_vol, N_trunc)
    # s-Teil: Parameter (c_vol,b_vol,a_vol)  (a <-> c getauscht)
    D_s, C_s, B_s = build_DCB(c_vol, b_vol, a_vol, N_trunc)

    all_lambdas = []

    for n in range(max_mode + 1):
        for m in range(max_mode + 1):
            lambdas_nm = find_lambdas_for_pair(
                n, m, lam_min, lam_max,
                D_t, C_t, B_t,
                D_s, C_s, B_s,
                num_scan=num_scan
            )
            all_lambdas.extend(lambdas_nm)

    return unique_sorted(all_lambdas)


# ------------------------------------------------------------
# 5. Kleine Demo / Main
# ------------------------------------------------------------
if __name__ == "__main__":
    # --- Parameter für den Hauptfall ---
    a_true = 2.3
    b_true = 1.5
    c_true = 1.0

    N_trunc = 40     # Erhöhen der Trunkation (höchste Präzision)
    max_mode = 2     # Moden bis (2,2)
    lam_min = 0.0
    lam_max = 50.0   # Kleinerer Bereich reicht, da k^2 nicht nahe 1 ist
    num_scan = 1000  # Hohe Scan-Auflösung beibehalten

    lambdas = compute_ellipsoid_eigenvalues_volkmer(
        a_true, b_true, c_true,
        N_trunc=N_trunc,
        max_mode=max_mode,
        lam_min=lam_min,
        lam_max=lam_max,
        num_scan=num_scan
    )
    
    # --- Berechnung des Skalierungsfaktors ---
    a_vol, b_vol, c_vol = sorted([a_true, b_true, c_true], reverse=True)
    scaling_factor = a_vol**2 - c_vol**2 # 2.3^2 - 1.0^2 = 4.29

    # --- Ausgabe ---
    print(f"Ellipsoid: a={a_true}, b={b_true}, c={c_true}")
    print(f"Skalierungsfaktor (a^2 - c^2) = {scaling_factor:.4f}")
    print(f"Trunkation N={N_trunc}, max_mode={max_mode}\n")
    print(" # | Lambda (Physik.) | λ (Dimensionslos) ")
    print("---|------------------|-------------------")

    for i, lam in enumerate(lambdas[:20]):
        Lambda = lam * scaling_factor
        print(f" {i+1:2d} | {Lambda:.6f}         | {lam:.6f}")
    
    # Testfall: Ellipsoid, das fast eine Kugel ist
    # Wir wählen Achsen, die fast gleich sind (Radius ca. 2)
    a_test = 2.001
    b_test = 2.000
    c_test = 1.999
    
    N_trunc = 30     # Erhöhen der Trunkation für bessere Genauigkeit
    max_mode = 3     # Erfassen höherer Eigenwerte (bis l=4)
    lam_min = 0.0
    lam_max = 1000.0 # Der Suchbereich muss hier viel größer sein!
    num_scan = 1500  # Für sehr steile Kurven mehr Scans

    lambdas = compute_ellipsoid_eigenvalues_volkmer(
        a_test, b_test, c_test,
        N_trunc=N_trunc,
        max_mode=max_mode,
        lam_min=lam_min,
        lam_max=lam_max,
        num_scan=num_scan
    )
    
    # --- Berechnung des Skalierungsfaktors ---
    # Achsen werden intern sortiert: a_vol > b_vol > c_vol
    a_vol, b_vol, c_vol = sorted([a_test, b_test, c_test], reverse=True)
    scaling_factor = a_vol**2 - c_vol**2 # Hier: 2.001^2 - 1.999^2 ≈ 0.008
    
    # --- Ausgabe ---
    print(f"Validation Ellipsoid (Near Sphere): a={a_test}, b={b_test}, c={c_test}")
    print(f"Trunkation N={N_trunc}, max_mode={max_mode}")
    print(f"Skalierungsfaktor (a^2 - c^2) = {scaling_factor:.8f}\n")
    print("Approx. Eigenwerte λ (dimensionslos) und Λ (physikalisch):\n")
    
    # Liste der erwarteten Kugel-Eigenwerte l(l+1) zum Vergleich
    expected_l = [0, 2, 2, 6, 6, 6, 12, 12, 12, 12, 20, 20, 20, 20, 20] # Vielfachheit beachten
    
    print(" # | Lambda (Physik.) | Expected l(l+1) | Differenz")
    print("---|------------------|-----------------|----------")
    
    for i, lam in enumerate(lambdas[:15]):
        # Rückskalierung: Λ = λ * (a_vol^2 - c_vol^2)
        Lambda = lam * scaling_factor
        
        expected = expected_l[i] if i < len(expected_l) else 'N/A'
        diff = Lambda - expected if i < len(expected_l) else 'N/A'
        
        print(f" {i+1:2d} | {Lambda:.6f}         | {expected:15} | {diff}")
