# Working Capital Simulator

Proyecto de Machine Learning para análisis de gestión de capital de trabajo (working capital). Agrupa empresas por su perfil operativo, predice niveles óptimos de inventario / AR / AP, y permite simular ajustes en DIO, DSO y DPO con un simulador interactivo.

## Estructura

| Archivo | Descripción |
|---|---|
| `app.py` | App Streamlit con el simulador interactivo |
| `finanzas_final.ipynb` | Notebook completo: descarga, clustering, módulos 2 y 3 |
| `dataset_finanzas_real.csv` | Dataset de 96 observaciones reales (24 empresas × 4 años) |
| `guia_finanzas.html` | Guía de estudio del proyecto (conceptos y defensa) |
| `Instrucciones.md` | Instrucciones originales del proyecto |
| `requirements.txt` | Dependencias Python |

## Cómo correr la app localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre en `http://localhost:8501`.

## Cómo correr el notebook

```bash
pip install -r requirements.txt jupyter yfinance matplotlib seaborn ipywidgets
jupyter notebook finanzas_final.ipynb
```

## Pipeline en 13 pasos

1. Configurar tickers (25 empresas en 7 sectores).
2. Descargar estados financieros desde Yahoo Finance (yfinance) — últimos 4 años fiscales, sin imputación.
3. Calcular ratios: DIO, DSO, DPO, CCC + porcentajes de activo corriente.
4. EDA con boxplots para detectar outliers.
5. Winsorización al 2.5% por cola.
6. Escalado con RobustScaler.
7. Selección de k con 4 métricas (inercia, silhouette, Davies-Bouldin, Calinski-Harabasz).
8. K-means con k=6.
9. PCA con 3 componentes para visualización (85.8% varianza acumulada).
10. Validación cruzada con clustering jerárquico Ward → ARI = 0.841.
11. Perfil de clusters e identificación del cluster eficiente (menor CCC promedio).
12. Módulo 2: targets por cluster como mediana del top-30% por menor CCC.
13. Módulo 3: simulador interactivo (Streamlit).

## Marco teórico

Cash Conversion Cycle (CCC) = DIO + DSO − DPO. Menor CCC = menos capital atrapado en la operación.

- **DIO** (Days Inventory Outstanding): días para vender el inventario.
- **DSO** (Days Sales Outstanding): días para cobrar al cliente.
- **DPO** (Days Payable Outstanding): días para pagar al proveedor (más alto = mejor).
