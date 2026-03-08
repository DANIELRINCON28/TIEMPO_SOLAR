"""
pdf_generator.py — Generación de reportes PDF para SolarMotion.
Usa fpdf2 para crear un PDF con datos, métricas, ecuaciones y gráficas.
"""

import io
import os
import tempfile
from datetime import datetime

from fpdf import FPDF


def _safe(text: str) -> str:
    """Reemplaza caracteres fuera del rango Latin-1 con equivalentes ASCII seguros."""
    replacements = {
        "\u2014": " - ",   # em dash —
        "\u2013": " - ",   # en dash –
        "\u00b7": "*",     # middle dot ·
        "\u00b2": "2",     # superscript 2
        "\u00b3": "3",     # superscript 3
        "\u03b1": "alpha",
        "\u03b2": "beta",
        "\u03b5": "epsilon",
        "\u221e": "inf",
        "\u00d7": "x",
        "\u00f7": "/",
        "\u2248": "~=",
        "\u2260": "!=",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u00b0": "deg",
        "\u00b1": "+/-",
        "\u00b5": "u",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Remove any remaining non-Latin-1 characters
    return text.encode("latin-1", errors="replace").decode("latin-1")


class SolarPDF(FPDF):
    """PDF personalizado con encabezado y pie de página."""

    def __init__(self, mode: str, model_name: str):
        super().__init__()
        self.mode = mode
        self.model_name = model_name
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, "SolarMotion Tracker & Modeler", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, _safe(f"Reporte de Analisis - {self.mode} - {self.model_name}"), align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 5, f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        # Línea separadora
        self.set_draw_color(79, 70, 229)  # indigo
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 10, f"Página {self.page_no()}/{{nb}}", align="C")


def generate_pdf(
    mode: str,
    model_name: str,
    data_x,
    data_y,
    metrics: dict,
    equation: str,
    plot_image_bytes: bytes | None = None,
    derivative_image_bytes: bytes | None = None,
    evidence_images: list | None = None,
    notes: str = "",
) -> bytes:
    """
    Genera un reporte PDF completo y devuelve los bytes del archivo.

    Parameters
    ----------
    mode : str  —  "Intensidad Lumínica" o "Rastreo de Sombras"
    model_name : str
    data_x, data_y : list/array
    metrics : dict con MSE, RMSE, R2
    equation : str
    plot_image_bytes : PNG bytes de la gráfica principal
    derivative_image_bytes : PNG bytes de la gráfica de derivada (opcional)
    evidence_images : lista de (nombre, bytes) de imágenes de evidencia
    notes : str  —  observaciones del usuario

    Returns
    -------
    bytes del PDF generado.
    """
    pdf = SolarPDF(mode=mode, model_name=model_name)
    pdf.alias_nb_pages()
    pdf.add_page()

    # --- Sección 1: Información del experimento ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "1. Informacion del Experimento", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    info_lines = [
        f"Modo: {_safe(mode)}",
        f"Modelo seleccionado: {_safe(model_name)}",
        f"Puntos de datos: {len(data_x)}",
    ]
    for line in info_lines:
        pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Sección 2: Métricas ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "2. Metricas del Modelo", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Tabla de metricas
    pdf.set_font("Helvetica", "B", 10)
    col_w = 60
    pdf.set_fill_color(79, 70, 229)
    pdf.set_text_color(255, 255, 255)
    for h in ["Metrica", "Valor", "Descripcion"]:
        pdf.cell(col_w, 8, h, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    rows = [
        ("MSE", f"{metrics.get('MSE', 0):.6f}", "Error cuadratico medio"),
        ("RMSE", f"{metrics.get('RMSE', 0):.6f}", "Raiz del error cuadratico medio"),
        ("R2", f"{metrics.get('R2', 0):.6f}", "Coeficiente de determinacion"),
    ]
    for r in rows:
        for val in r:
            pdf.cell(col_w, 7, str(val), border=1, align="C")
        pdf.ln()
    pdf.ln(4)

    # --- Sección 3: Ecuación ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "3. Ecuacion del Modelo", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Courier", "B", 12)
    pdf.set_text_color(79, 70, 229)
    # Manejar ecuaciones largas — sanitizar caracteres especiales
    pdf.multi_cell(0, 7, _safe(equation))
    pdf.ln(4)

    # --- Sección 4: Datos ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "4. Tabla de Datos", new_x="LMARGIN", new_y="NEXT")  # ASCII safe
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(79, 70, 229)
    pdf.set_text_color(255, 255, 255)
    x_label = "Hora (X)"
    y_label = "Intensidad (Lux)" if "Luz" in mode or "Intensidad" in mode else "Long. sombra (cm)"
    pdf.cell(90, 8, x_label, border=1, fill=True, align="C")
    pdf.cell(90, 8, y_label, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    for xi, yi in zip(data_x, data_y):
        pdf.cell(90, 6, str(round(float(xi), 4)), border=1, align="C")
        pdf.cell(90, 6, str(round(float(yi), 4)), border=1, align="C")
        pdf.ln()
    pdf.ln(4)

    # --- Sección 5: Gráficas ---
    tmp_files = []
    if plot_image_bytes:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, "5. Grafica Principal (Datos vs Modelo)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(plot_image_bytes)
        tmp.flush()
        tmp_files.append(tmp.name)
        pdf.image(tmp.name, x=10, w=190)
        tmp.close()
        pdf.ln(6)

    if derivative_image_bytes:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, "6. Grafica de Velocidad (Derivada)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(derivative_image_bytes)
        tmp.flush()
        tmp_files.append(tmp.name)
        pdf.image(tmp.name, x=10, w=190)
        tmp.close()
        pdf.ln(6)

    # --- Sección: Evidencias ---
    if evidence_images:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 30, 30)
        section_n = 7 if derivative_image_bytes else 6
        pdf.cell(0, 8, f"{section_n}. Evidencias Fotograficas", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        for name, img_bytes in evidence_images:
            try:
                ext = os.path.splitext(name)[1] or ".png"
                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                tmp.write(img_bytes)
                tmp.flush()
                tmp_files.append(tmp.name)
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 6, _safe(name), new_x="LMARGIN", new_y="NEXT")
                pdf.image(tmp.name, x=10, w=120)
                tmp.close()
                pdf.ln(4)
            except Exception:
                pass

    # --- Notas ---
    if notes and notes.strip():
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, "Observaciones", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 6, _safe(notes))

    # Generar bytes
    pdf_bytes = pdf.output()

    # Limpiar archivos temporales
    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    return bytes(pdf_bytes)
