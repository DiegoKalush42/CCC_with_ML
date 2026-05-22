INSTRUCCIÓN PARA EL LLM:
Quiero que me ayudes a construir un proyecto de finanzas que usa Machine Learning para administrar los activos de corto plazo de una empresa, comparándola contra empresas similares. El marco teórico es la gestión de capital de trabajo (working capital): ciclo de efectivo, costos de mantener (carrying) vs. costos de faltante (shortage). Necesito código en Python (pandas, scikit-learn, matplotlib) y explicaciones claras. El proyecto tiene tres módulos:
MÓDULO 1 — Clustering de empresas comparables

Objetivo: agrupar empresas por su perfil de gestión de capital de trabajo (no por sector ni nombre).
Datos de entrada: un dataset de empresas (≥20–30) con ventas, costo de ventas, inventario, cuentas por cobrar, cuentas por pagar y activos totales.
Convierte los datos en features comparables y normalizadas: DIO (días de inventario), DSO (días de cobro), DPO (días de pago), Cash Conversion Cycle (CCC), e inventario/CxC/efectivo como % de ventas.
Usa K-means como método principal; elige el número de grupos (k) con el método de silueta o del codo.
Genera además un dendrograma con clustering jerárquico aglomerativo para validar visualmente los grupos.
No uses DBSCAN (deja empresas sin grupo) ni redes neuronales (sobreajuste con dataset pequeño, poca interpretabilidad).
Salida: cada empresa asignada a su cluster + el perfil promedio (centroide) de cada grupo.

MÓDULO 2 — Predicción del nivel óptimo de activos de corto plazo

Para una empresa dada, estima su nivel objetivo (target) de activos de corto plazo dentro de su cluster.
Usa un enfoque híbrido: entrena una regresión usando SOLO las mejores empresas del cluster (las del top por menor CCC o mayor rentabilidad sobre capital de trabajo). Así el target refleja la mejor práctica y a la vez se personaliza al tamaño/perfil de la empresa.
Deja también disponible un benchmark simple (promedio del top del cluster) como método de respaldo.
Salida: el nivel óptimo (target) y la brecha vs. el nivel actual (ej. "inventario actual 100, óptimo 75 → sobran 25").

MÓDULO 3 — Simulador de sensibilidad ("qué pasa si")

Construye un módulo interactivo donde el usuario ajuste tres palancas: DIO, DSO y DPO.
Recalcula en tiempo real:

Cash Conversion Cycle = DIO + DSO − DPO
Efectivo liberado = nivel actual − target


Permite ajustar varias palancas a la vez y muestra el efecto combinado.
Salida: CCC y efectivo liberado antes vs. después, más la receta de ajustes que cierra la brecha al óptimo (ej. "baja DIO a 60 y DSO a 110").

Extras:

Comenta el código para que sea explicable.
Incluye una breve justificación de por qué se eligió cada método.