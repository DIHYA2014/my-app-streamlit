"""TP Machine Learning : régression linéaire et Perceptron (application Streamlit)."""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split

from contenu import CODE_TP1, CODE_TP2, NOTE_TP1, QA

A, B, INK = "#1f4fd8", "#d6336c", "#16213a"
st.set_page_config(page_title="TP Machine Learning", page_icon="📈", layout="centered")
st.session_state.setdefault("seed1", 42)
st.session_state.setdefault("seed2", 42)


@st.cache_data
def charger(nom):
    fichier = Path(__file__).resolve().parent / f"{nom}.csv"
    if fichier.exists():
        return pd.read_csv(fichier)
    d = load_iris(as_frame=True) if nom == "iris" else load_diabetes(as_frame=True)
    return d.frame


def metriques(y, p):
    mse = mean_squared_error(y, p)
    return {"MAE": mean_absolute_error(y, p), "MSE": mse,
            "RMSE": mse ** 0.5, "R² test": r2_score(y, p)}


def perceptron(X, y, eta, epochs, aleatoire=False, seed=0, n_log=8):
    """Perceptron from scratch. Retourne poids, biais, erreurs par époque et journal."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.5, X.shape[1]) if aleatoire else np.zeros(X.shape[1])
    b = float(rng.normal(0, 0.5)) if aleatoire else 0.0
    erreurs, journal = [], []
    for e in range(epochs):
        err = 0
        for xi, yi in zip(X, y):
            z = float(w @ xi + b)
            yh = 1 if z >= 0 else 0
            d = int(yi) - yh
            if d != 0:
                err += 1
                if len(journal) < n_log:
                    nw = w + eta * d * xi
                    journal.append(f"Époque {e+1} : z={z:.2f} → ŷ={yh}, y={int(yi)}, erreur={d} "
                                   f"⇒ w=({nw[0]:.2f}, {nw[1]:.2f}), b={b + eta * d:.2f}")
            w = w + eta * d * xi
            b += eta * d
        erreurs.append(err)
    return w, b, erreurs, journal


st.title("TP Machine Learning : voir les modèles apprendre")
st.caption("Régression linéaire (TP1) et Perceptron (TP2). Chaque partie explique, puis te laisse expérimenter.")
t_home, t1, t2, t_qa, t_code = st.tabs(["Démarrer", "TP1 Régression", "TP2 Perceptron", "Questions", "Code Python"])

# ------------------------------------------------------------------ Démarrer
with t_home:
    st.header("Comment utiliser ce site")
    st.info("Ce site reprend les deux TP du module ML. Les expériences utilisent les **vrais jeux de données de "
            "Scikit-learn** (Iris et Diabetes), chargés automatiquement : rien à téléverser. Pour le rendu, "
            "refais les mêmes étapes dans ton notebook : l’onglet « Code Python » te donne le code.")
    st.subheader("Plan de travail conseillé")
    st.markdown(
        "1. Lis l’explication de TP1, puis déplace le curseur de découpage et relance le tirage : vois comment R² change.\n"
        "2. Passe à TP2 : règle η et le nombre d’époques, puis suis le journal des mises à jour.\n"
        "3. Réponds aux questions du sujet dans l’onglet « Questions », avec tes propres mots dans le rapport.\n"
        "4. Copie le code, exécute-le dans Colab, ajoute tes graphiques et rédige le rapport de 3 à 5 pages.")
    st.caption("Rappel du sujet : rendu = notebook .ipynb + rapport PDF. Ce site t’aide à comprendre, "
               "il ne remplace pas ces livrables. Le code doit être commenté.")

# ------------------------------------------------------------------ TP1
with t1:
    st.header("TP1 : régression linéaire")
    st.subheader("L’idée")
    st.markdown("La régression linéaire prédit une valeur **continue** comme une somme pondérée des variables d’entrée. "
                "Apprendre = trouver les poids qui minimisent l’erreur quadratique moyenne sur les données d’entraînement.")
    st.markdown("**Simple** : ŷ = w·x + b  \n**Multiple** : ŷ = w₁x₁ + w₂x₂ + … + wₙxₙ + b")
    st.markdown("**Pente (w)** : de combien ŷ change quand x augmente de 1. **Ordonnée à l’origine (b)** : la valeur de ŷ "
                "quand toutes les variables valent 0. Les poids se calculent avec les équations normales "
                "w = (XᵀX)⁻¹Xᵀy, ce que calcule `LinearRegression` de Scikit-learn, utilisé ici.")

    st.subheader("Expérience")
    c = dict(d="diabetes", y="target", s="bmi", m=None)
    df = charger("diabetes")
    multi = [k for k in df.columns if k != c["y"]]
    k1, k2 = st.columns([3, 1], vertical_alignment="bottom")
    if k2.button("Nouveau tirage", key="re1"):
        st.session_state.seed1 = int(np.random.randint(0, 10**6))
    part = k1.slider("Part d’entraînement (%)", 50, 90, 80, 5)
    seed = st.session_state.seed1

    i_tr, i_te = train_test_split(np.arange(len(df)), train_size=part / 100, random_state=seed)
    y = df[c["y"]].values
    mods, rows = {}, {}
    for nom, cols in [("SLR (1 variable)", [c["s"]]), (f"MLR ({len(multi)} variables)", multi)]:
        X = df[cols].values
        mod = LinearRegression().fit(X[i_tr], y[i_tr])
        mods[nom] = mod
        r = metriques(y[i_te], mod.predict(X[i_te]))
        r["R² entraînement"] = r2_score(y[i_tr], mod.predict(X[i_tr]))
        rows[nom] = r

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = df[c["s"]].values
    ax.scatter(x[i_tr], y[i_tr], s=18, alpha=.55, color=A, label="entraînement")
    ax.scatter(x[i_te], y[i_te], s=26, alpha=.9, color=B, label="test")
    xs = np.linspace(x.min(), x.max(), 50).reshape(-1, 1)
    ax.plot(xs, mods["SLR (1 variable)"].predict(xs), color=INK, lw=2.5, label="modèle simple")
    ax.set_xlabel(c["s"]); ax.set_ylabel(c["y"]); ax.grid(alpha=.25); ax.legend(fontsize=8)
    st.pyplot(fig); plt.close(fig)

    st.table(pd.DataFrame(rows).round(3))
    ws, wm = mods["SLR (1 variable)"], list(mods.values())[1]
    terme = " + ".join(f"{w:.2f} × {n}" for w, n in zip(wm.coef_, multi))
    st.markdown(f"**Modèle simple appris** : {c['y']} = {ws.coef_[0]:.2f} × {c['s']} + {ws.intercept_:.2f}  \n"
                f"**Modèle multiple appris** : {c['y']} = {terme} + {wm.intercept_:.2f}")
    st.caption(f"{len(i_tr)} échantillons d’entraînement, {len(i_te)} de test (graine {seed}).")

    st.subheader("Les métriques")
    st.markdown("- **MAE** : erreur moyenne en valeur absolue, dans l’unité de y, facile à lire.\n"
                "- **MSE** : moyenne des carrés des erreurs, elle punit fortement les grosses erreurs.\n"
                "- **RMSE** : racine du MSE, de nouveau dans l’unité de y.\n"
                "- **R²** : part de la variance de y expliquée par le modèle (1 = parfait, 0 = pas mieux que la moyenne).")
    st.markdown("Observe : le modèle multiple utilise plusieurs variables, et son R² de test dépasse presque toujours celui "
                "du modèle simple. Mais plus de variables ne garantit pas mieux : sur peu de données, on peut surapprendre. "
                "Compare toujours **sur le jeu de test**.")

# ------------------------------------------------------------------ TP2
with t2:
    st.header("TP2 : le Perceptron")
    st.subheader("Le neurone artificiel")
    st.markdown("Le Perceptron calcule une somme pondérée, puis applique une fonction seuil (échelon).")
    st.markdown("**z = w₁x₁ + w₂x₂ + b**  \n**ŷ = 1 si z ≥ 0, sinon 0**  \n**wᵢ ← wᵢ + η (y − ŷ) xᵢ**  et  **b ← b + η (y − ŷ)**")
    st.markdown("Si la prédiction est juste, y − ŷ = 0 et rien ne change. Si elle est fausse, le point « tire » la droite "
                "de séparation vers lui. La frontière est la droite w₁x₁ + w₂x₂ + b = 0.")

    st.subheader("Expérience : Setosa (0) contre Versicolor (1)")
    st.caption("Variables : longueur du pétale (axe x) et largeur du pétale (axe y). Données : Iris de Scikit-learn.")
    iris = charger("iris")
    sub = iris[iris["target"] < 2]
    X2 = sub[["petal length (cm)", "petal width (cm)"]].values
    y2 = sub["target"].values

    p1, p2 = st.columns(2)
    eta = p1.slider("Taux d’apprentissage η", 0.01, 1.0, 0.10, 0.01)
    ep = p2.slider("Époques", 1, 50, 10)
    q1, q2 = st.columns([2, 1], vertical_alignment="center")
    alea = q1.checkbox("Initialisation aléatoire des poids (sinon w = 0 et b = 0)")
    if q2.button("Nouveau tirage", key="re2"):
        st.session_state.seed2 = int(np.random.randint(0, 10**6))
    seed2 = st.session_state.seed2

    Xtr, Xte, ytr, yte = train_test_split(X2, y2, test_size=0.2, random_state=seed2, stratify=y2)
    w, b, errs, log = perceptron(Xtr, ytr, eta, ep, alea, seed2)

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.scatter(*Xtr[ytr == 0].T, s=18, alpha=.6, color=A, label="Setosa")
    ax.scatter(*Xtr[ytr == 1].T, s=18, alpha=.6, color=B, label="Versicolor")
    ax.scatter(*Xte[yte == 0].T, s=70, facecolors="none", edgecolors=A, lw=1.8)
    ax.scatter(*Xte[yte == 1].T, s=70, facecolors="none", edgecolors=B, lw=1.8, label="cercle = point de test")
    x0, x1, y0, y1 = 0.5, 7.2, 0.0, 2.8
    if abs(w[1]) > 1e-9:
        ax.plot([x0, x1], [-(b + w[0] * x0) / w[1], -(b + w[0] * x1) / w[1]], color=INK, lw=2.5)
    elif abs(w[0]) > 1e-9:
        ax.axvline(-b / w[0], color=INK, lw=2.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("Longueur du pétale (cm)"); ax.set_ylabel("Largeur du pétale (cm)"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    st.pyplot(fig); plt.close(fig)

    st.subheader("Courbe d’erreur d’entraînement")
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    ax.plot(range(1, len(errs) + 1), errs, color=B, lw=2.5, marker="o", ms=4)
    ax.set_xlabel("Époque"); ax.set_ylabel("Erreurs"); ax.set_ylim(bottom=0); ax.grid(alpha=.25)
    st.pyplot(fig); plt.close(fig)

    yp = ((Xte @ w + b) >= 0).astype(int)
    cm = confusion_matrix(yte, yp, labels=[0, 1])
    st.table(pd.DataFrame(cm, index=["Réel 0 (Setosa)", "Réel 1 (Versicolor)"], columns=["Prédit 0", "Prédit 1"]))
    st.markdown(f"Accuracy **{accuracy_score(yte, yp):.3f}** · Précision **{precision_score(yte, yp, zero_division=0):.3f}** · "
                f"Rappel **{recall_score(yte, yp, zero_division=0):.3f}** · F1 **{f1_score(yte, yp, zero_division=0):.3f}**")
    st.caption(f"Poids : w₁={w[0]:.3f}, w₂={w[1]:.3f}, b={b:.3f}. Positif = Versicolor. Test sur {len(yte)} échantillons.")

    st.subheader("Journal des premières mises à jour")
    st.code("\n".join(log) if log else "Aucune erreur : le modèle a classé tous les points d’entraînement dès la première époque.",
            language=None)

    st.subheader("Multiclasse (partie C)")
    st.markdown("Avec 3 classes, on entraîne 3 Perceptrons « une classe contre le reste » (One-vs-Rest) : Setosa vs reste, "
                "Versicolor vs reste, Virginica vs reste. Pour prédire, on calcule z pour chacun et on garde la classe au z le "
                "plus grand. Attention : Versicolor et Virginica ne sont pas parfaitement séparables par une droite, donc un "
                "seul neurone fera des erreurs sur ces deux classes.")
    if st.checkbox("Lancer le One-vs-Rest sur les 3 classes (4 variables)"):
        o1, o2 = st.columns(2)
        ep3 = o1.slider("Époques (multiclasse)", 5, 200, 50, key="ep3")
        std = o2.checkbox("Standardiser les variables", value=True)
        X3, y3 = iris.drop(columns="target").values, iris["target"].values
        a, bb, c3, d3 = train_test_split(X3, y3, test_size=0.2, random_state=seed2, stratify=y3)
        if std:
            mu, sd = a.mean(0), a.std(0)
            a, bb = (a - mu) / sd, (bb - mu) / sd
        noms = ["setosa", "versicolor", "virginica"]
        ms = [perceptron(a, (c3 == k).astype(int), eta, ep3, alea, seed2) for k in range(3)]
        Z = np.column_stack([bb @ m[0] + m[1] for m in ms])
        yp3 = Z.argmax(axis=1)
        st.markdown(f"Accuracy : **{accuracy_score(d3, yp3):.3f}**")
        st.table(pd.DataFrame(confusion_matrix(d3, yp3), index=[f"Réel {n}" for n in noms],
                              columns=[f"Prédit {n}" for n in noms]))
        st.caption("Erreurs d’entraînement à la dernière époque : " +
                   ", ".join(f"{n} vs reste = {m[2][-1]}" for n, m in zip(noms, ms)))

# ------------------------------------------------------------------ Questions
with t_qa:
    st.header("Questions du sujet")
    st.caption("Réponses courtes pour t’aider à comprendre. Reformule-les dans ton rapport, avec tes résultats.")
    for titre, items in QA:
        st.subheader(titre)
        for q, r in items:
            with st.expander(q):
                st.markdown(r)

# ------------------------------------------------------------------ Code
with t_code:
    st.header("Code Python à adapter dans Colab")
    st.subheader("TP1 : régression (Diabetes)")
    st.code(CODE_TP1, language="python")
    st.caption(NOTE_TP1)
    st.subheader("TP2 : Perceptron from scratch")
    st.code(CODE_TP2, language="python")
