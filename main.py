# ═══════════════════════════════════════════════════════════════════════════
# Reto 1: Humano vs Máquina - La Batalla de los Pingüinos 🐧
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
np.random.seed(42)

COLORES_ESPECIES = {
    'Adelie': '#FF6B6B',
    'Chinstrap': '#4ECDC4',
    'Gentoo': '#45B7D1'
}

print("╔═══════════════════════════════════════════════════════════════╗")
print("║     🐧 ESTACIÓN PALMER - SISTEMA DE CLASIFICACIÓN v2.0 🐧     ║")
print("╚═══════════════════════════════════════════════════════════════╝")

# ═══════════════════════════════════════════════════════════════════════════
# CARGAR DATOS
# ═══════════════════════════════════════════════════════════════════════════

df_original = sns.load_dataset('penguins')
df = df_original.dropna().reset_index(drop=True)

print(f"\n📊 Registros cargados: {len(df)}")
print(f"🏝️  Islas: {df['island'].nunique()}")
print(f"🐧 Especies: {df['species'].nunique()}")

# ═══════════════════════════════════════════════════════════════════════════
# EXPLORACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

print("\n🔍 Estadísticas descriptivas:")
print(df.describe())

print(f"\nRango de masa corporal: {df['body_mass_g'].min()} - {df['body_mass_g'].max()} g")
print(f"Longitud promedio de pico: {df['bill_length_mm'].mean():.2f} mm")
print(f"Longitud promedio de aleta: {df['flipper_length_mm'].mean():.2f} mm")

columnas_numericas = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
promedios_por_especie = df.groupby('species')[columnas_numericas].mean()
print("\nPromedio por especie:")
print(promedios_por_especie)

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
variables = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
titulos = ['Longitud del Pico (mm)', 'Profundidad del Pico (mm)',
           'Longitud de Aleta (mm)', 'Masa Corporal (g)']

for ax, var, titulo in zip(axes.flat, variables, titulos):
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        ax.hist(subset[var], alpha=0.6, label=species,
                color=COLORES_ESPECIES[species], bins=20, edgecolor='white')
    ax.set_xlabel(titulo)
    ax.set_ylabel('Frecuencia')
    ax.legend()
    ax.set_title(f'Distribución de {titulo} por Especie')

plt.tight_layout()
plt.savefig('pinguinos_distribucion.png', dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for species in df['species'].unique():
    subset = df[df['species'] == species]
    axes[0].scatter(subset['bill_length_mm'], subset['bill_depth_mm'],
                   c=COLORES_ESPECIES[species], label=species, alpha=0.7, s=60)
axes[0].set_xlabel('Longitud del Pico (mm)')
axes[0].set_ylabel('Profundidad del Pico (mm)')
axes[0].set_title('Dimensiones del Pico por Especie')
axes[0].legend()

for species in df['species'].unique():
    subset = df[df['species'] == species]
    axes[1].scatter(subset['flipper_length_mm'], subset['body_mass_g'],
                   c=COLORES_ESPECIES[species], label=species, alpha=0.7, s=60)
axes[1].set_xlabel('Longitud de Aleta (mm)')
axes[1].set_ylabel('Masa Corporal (g)')
axes[1].set_title('Aleta vs Masa por Especie')
axes[1].legend()

plt.tight_layout()
plt.savefig('pinguinos_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# CLASIFICADOR HUMANO v1
# ═══════════════════════════════════════════════════════════════════════════

def clasificador_humano(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    """Clasifica un pingüino usando reglas diseñadas por un humano."""
    if flipper_length_mm >= 205 and bill_depth_mm <= 17:
        return "Gentoo"
    elif bill_length_mm >= 45 and bill_depth_mm >= 17:
        return "Chinstrap"
    else:
        return "Adelie"


df['prediccion_humana'] = df.apply(
    lambda fila: clasificador_humano(
        fila['bill_length_mm'], fila['bill_depth_mm'],
        fila['flipper_length_mm'], fila['body_mass_g']
    ), axis=1
)

accuracy_general = (df['species'] == df['prediccion_humana']).mean()
print(f"\nAccuracy del clasificador humano (dataset completo): {accuracy_general:.2%}")

# ═══════════════════════════════════════════════════════════════════════════
# DIVISIÓN TRAIN / TEST
# ═══════════════════════════════════════════════════════════════════════════

X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 División: {len(X_train)} entrenamiento / {len(X_test)} prueba")

# ═══════════════════════════════════════════════════════════════════════════
# EVALUACIÓN CLASIFICADOR HUMANO
# ═══════════════════════════════════════════════════════════════════════════

predicciones_humano = [
    clasificador_humano(row['bill_length_mm'], row['bill_depth_mm'],
                        row['flipper_length_mm'], row['body_mass_g'])
    for _, row in X_test.iterrows()
]

accuracy_humano = accuracy_score(y_test, predicciones_humano)
print(f"\n🧠 Accuracy Humano: {accuracy_humano:.2%} ({int(accuracy_humano * len(y_test))}/{len(y_test)})")

cm_humano = confusion_matrix(y_test, predicciones_humano, labels=['Adelie', 'Chinstrap', 'Gentoo'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm_humano, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Adelie', 'Chinstrap', 'Gentoo'],
            yticklabels=['Adelie', 'Chinstrap', 'Gentoo'])
plt.xlabel('Predicción del Humano')
plt.ylabel('Especie Real')
plt.title('Matriz de Confusión - Clasificador Humano')
plt.tight_layout()
plt.savefig('confusion_humano.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Reporte Humano:")
print(classification_report(y_test, predicciones_humano))

# ═══════════════════════════════════════════════════════════════════════════
# MODELO MACHINE LEARNING
# ═══════════════════════════════════════════════════════════════════════════

modelo_ml = DecisionTreeClassifier(random_state=42)
modelo_ml.fit(X_train, y_train)

print(f"\n✅ Árbol entrenado | Profundidad: {modelo_ml.get_depth()} | Hojas: {modelo_ml.get_n_leaves()}")

predicciones_ml = modelo_ml.predict(X_test)
accuracy_ml = accuracy_score(y_test, predicciones_ml)
print(f"🤖 Accuracy ML: {accuracy_ml:.2%} ({int(accuracy_ml * len(y_test))}/{len(y_test)})")

cm_ml = confusion_matrix(y_test, predicciones_ml, labels=['Adelie', 'Chinstrap', 'Gentoo'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Adelie', 'Chinstrap', 'Gentoo'],
            yticklabels=['Adelie', 'Chinstrap', 'Gentoo'])
plt.xlabel('Predicción ML')
plt.ylabel('Especie Real')
plt.title('Matriz de Confusión - Modelo ML')
plt.tight_layout()
plt.savefig('confusion_ml.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Reporte ML:")
print(classification_report(y_test, predicciones_ml))

plt.figure(figsize=(20, 10))
plot_tree(modelo_ml,
          feature_names=['bill_length', 'bill_depth', 'flipper_length', 'body_mass'],
          class_names=['Adelie', 'Chinstrap', 'Gentoo'],
          filled=True, rounded=True, fontsize=10)
plt.title('Árbol de Decisión', fontsize=16)
plt.tight_layout()
plt.savefig('arbol_pinguinos.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# COMPARACIÓN FINAL
# ═══════════════════════════════════════════════════════════════════════════

diferencia = accuracy_ml - accuracy_humano
print("\n⚔️  BATALLA FINAL: HUMANO VS MÁQUINA")
print(f"🧠 Humano: {accuracy_humano:.2%}")
print(f"🤖 ML:     {accuracy_ml:.2%}")
if diferencia > 0:
    print(f"🏆 Ganador: LA MÁQUINA (+{diferencia:.2%})")
elif diferencia < 0:
    print(f"🏆 Ganador: EL HUMANO (+{-diferencia:.2%})")
else:
    print("🤝 Empate técnico")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
bars = axes[0].bar(['Humano', 'Máquina (ML)'],
                   [accuracy_humano * 100, accuracy_ml * 100],
                   color=['#FF6B6B', '#4ECDC4'], edgecolor='white', linewidth=2)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Comparación de Accuracy', fontweight='bold')
axes[0].set_ylim(0, 105)
for bar, acc in zip(bars, [accuracy_humano * 100, accuracy_ml * 100]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

sns.heatmap(cm_humano, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['A', 'C', 'G'], yticklabels=['A', 'C', 'G'])
axes[1].set_title('Matriz Humano', fontweight='bold')

sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['A', 'C', 'G'], yticklabels=['A', 'C', 'G'])
axes[2].set_title('Matriz ML', fontweight='bold')

plt.suptitle('🐧 Batalla de Pingüinos: Humano vs Máquina 🐧', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('batalla_final.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS DE ERRORES
# ═══════════════════════════════════════════════════════════════════════════

resultados = X_test.copy()
resultados['especie_real'] = y_test.values
resultados['pred_humano'] = predicciones_humano
resultados['pred_ml'] = predicciones_ml
resultados['error_humano'] = resultados['especie_real'] != resultados['pred_humano']
resultados['error_ml'] = resultados['especie_real'] != resultados['pred_ml']

errores_humano = resultados[resultados['error_humano']]
print(f"\n❌ Errores del HUMANO ({len(errores_humano)} casos):")
print(errores_humano[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                       'body_mass_g', 'especie_real', 'pred_humano']].to_string() if len(errores_humano) > 0 else "¡Ningún error! 🎉")

errores_ml = resultados[resultados['error_ml']]
print(f"\n❌ Errores del ML ({len(errores_ml)} casos):")
print(errores_ml[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                   'body_mass_g', 'especie_real', 'pred_ml']].to_string() if len(errores_ml) > 0 else "¡Ningún error! 🎉")

# ═══════════════════════════════════════════════════════════════════════════
# CLASIFICADOR HUMANO v2 (MEJORADO)
# ═══════════════════════════════════════════════════════════════════════════

def clasificador_humano_v2(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    """Clasificador mejorado basado en el árbol de decisión."""
    if flipper_length_mm > 207:
        if bill_depth_mm <= 17.65:
            return "Gentoo"
        elif body_mass_g > 4300:
            return "Gentoo"
        else:
            return "Chinstrap"
    else:
        if bill_length_mm <= 43:
            return "Adelie"
        else:
            return "Chinstrap"


predicciones_humano_v2 = [
    clasificador_humano_v2(row['bill_length_mm'], row['bill_depth_mm'],
                           row['flipper_length_mm'], row['body_mass_g'])
    for _, row in X_test.iterrows()
]

accuracy_humano_v2 = accuracy_score(y_test, predicciones_humano_v2)
print(f"\n📈 Clasificador v1: {accuracy_humano:.2%}")
print(f"📈 Clasificador v2: {accuracy_humano_v2:.2%}")
print(f"📈 Mejora: {(accuracy_humano_v2 - accuracy_humano):.2%}")
